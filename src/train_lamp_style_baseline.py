"""
Precise LaMP (Lanchantin, Sekhon, Qi — ECML 2019), adapted to use GraphSAGE
as the instance encoder.

Mechanism (matches paper Section 2.2):
  Label embeddings u_i^0 initialized from a learnable embedding matrix.
  For t = 1..T rounds:
    (a) Feature-to-Label message passing: attention from label embeddings
        (queries) to instance feature embedding (key/value) — updates labels
        with input-conditioned information.
    (b) Label-to-Label message passing: self-attention among label embeddings
        — updates labels with inter-label dependency information. Uses FULL
        attention (no predefined graph) — matches the paper's "structure-
        agnostic" default setting, distinct from ML-GCN's fixed graph.
  Readout: a shared linear layer maps each label's final embedding to one logit.

Necessary adaptation: the paper's inputs are naturally multi-token (e.g. words
in a document). GraphSAGE produces one embedding vector per protein, not
multiple tokens — used directly as the single feature-node for phase (a).

Usage:
    python src/train_lamp_style_baseline.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
import json
import os
import sys

sys.path.append(os.path.dirname(__file__))
from train_baseline import GraphSAGEEncoder, load_selected_labels, filter_labels
from metrics import compute_all_metrics, format_metrics


class LaMPHead(nn.Module):
    def __init__(self, instance_embed_dim, n_labels, label_dim=128, n_heads=4, n_rounds=2, n_feature_tokens=8):
        super().__init__()
        self.n_labels = n_labels
        self.n_rounds = n_rounds
        self.n_feature_tokens = n_feature_tokens
        self.label_dim = label_dim
        self.label_embeddings = nn.Parameter(torch.randn(n_labels, label_dim) * 0.1)
        # Project into MULTIPLE feature tokens, not one — with a single token,
        # attention is mathematically forced to produce identical output for
        # every label query (softmax over 1 key is trivially 1.0 regardless of
        # query) — this is the exact collapse Cursor's diagnostic found.
        self.feature_proj = nn.Linear(instance_embed_dim, n_feature_tokens * label_dim)

        self.feature_to_label_attn = nn.ModuleList(
            [nn.MultiheadAttention(label_dim, n_heads, batch_first=True) for _ in range(n_rounds)]
        )
        self.label_to_label_attn = nn.ModuleList(
            [nn.MultiheadAttention(label_dim, n_heads, batch_first=True) for _ in range(n_rounds)]
        )
        self.norm1 = nn.ModuleList([nn.LayerNorm(label_dim) for _ in range(n_rounds)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(label_dim) for _ in range(n_rounds)])
        # Feedforward blocks after each attention phase — WITHOUT these, stacked
        # self-attention layers collapse toward a shared representation (attention
        # rank collapse, Dong et al. 2021), which is what caused the stuck-at-zero
        # symptom: label embeddings converged toward each other, so the shared
        # readout produced nearly identical logits for every label.
        self.ffn1 = nn.ModuleList(
            [nn.Sequential(nn.Linear(label_dim, label_dim * 2), nn.ReLU(),
                            nn.Linear(label_dim * 2, label_dim)) for _ in range(n_rounds)]
        )
        self.ffn2 = nn.ModuleList(
            [nn.Sequential(nn.Linear(label_dim, label_dim * 2), nn.ReLU(),
                            nn.Linear(label_dim * 2, label_dim)) for _ in range(n_rounds)]
        )
        self.norm1b = nn.ModuleList([nn.LayerNorm(label_dim) for _ in range(n_rounds)])
        self.norm2b = nn.ModuleList([nn.LayerNorm(label_dim) for _ in range(n_rounds)])

        # LayerScale (Touvron et al. 2021): learnable, near-zero-initialized scale
        # on each sub-layer's output before adding to the residual. Without this,
        # the (initially uninformative, uniform) attention output grows large and
        # dominates the residual via LayerNorm's scale-invariance before attention
        # has had a chance to learn to differentiate labels — the exact collapse
        # Cursor's diagnostic traced (uniform attention entropy, |attn_out| >> |labels|).
        self.scale_attn1 = nn.ParameterList([nn.Parameter(torch.ones(label_dim) * 1e-3) for _ in range(n_rounds)])
        self.scale_ffn1 = nn.ParameterList([nn.Parameter(torch.ones(label_dim) * 1e-3) for _ in range(n_rounds)])
        self.scale_attn2 = nn.ParameterList([nn.Parameter(torch.ones(label_dim) * 1e-3) for _ in range(n_rounds)])
        self.scale_ffn2 = nn.ParameterList([nn.Parameter(torch.ones(label_dim) * 1e-3) for _ in range(n_rounds)])

        self.readout = nn.Linear(label_dim, 1)
        # Fix cold-start collapse: initialize bias to the true label rate's log-odds,
        # instead of 0. Without this, LayerNorm-normalized inputs + near-zero initial
        # weights produce near-0.5 probability for every label, which rounds to
        # all-negative under a 0.5 threshold and can take many epochs to escape.
        true_rate = 0.4175  # observed training label density (see earlier diagnostic)
        bias_init = torch.log(torch.tensor(true_rate / (1 - true_rate)))
        self.readout.bias.data.fill_(bias_init.item())

    def forward(self, z):
        N = z.shape[0]
        z_proj = self.feature_proj(z).view(N, self.n_feature_tokens, self.label_dim)  # (N, K, label_dim) — MULTIPLE tokens
        labels = self.label_embeddings.unsqueeze(0).expand(N, -1, -1)  # (N, n_labels, label_dim)

        for t in range(self.n_rounds):
            # (a) Feature-to-Label message passing
            attn_out, _ = self.feature_to_label_attn[t](labels, z_proj, z_proj)
            labels = self.norm1[t](labels + self.scale_attn1[t] * attn_out)
            labels = self.norm1b[t](labels + self.scale_ffn1[t] * self.ffn1[t](labels))

            # (b) Label-to-Label message passing (full attention, structure-agnostic)
            attn_out2, _ = self.label_to_label_attn[t](labels, labels, labels)
            labels = self.norm2[t](labels + self.scale_attn2[t] * attn_out2)
            labels = self.norm2b[t](labels + self.scale_ffn2[t] * self.ffn2[t](labels))

        logits = self.readout(labels).squeeze(-1)  # (N, n_labels)
        return logits


def train_epoch(encoder, head, loader, optimizer, device):
    encoder.train()
    head.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        z = encoder(batch.x, batch.edge_index)
        logits = head(z)
        loss = F.binary_cross_entropy_with_logits(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(encoder, head, loader, device):
    encoder.eval()
    head.eval()
    probs, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        z = encoder(batch.x, batch.edge_index)
        logits = head(z)
        probs.append(torch.sigmoid(logits).cpu())
        ys.append(batch.y.cpu())
    probs = torch.cat(probs, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()
    return compute_all_metrics(ys, probs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root = "./data/raw"
    selected_indices = load_selected_labels()
    train_dataset = filter_labels(PPI(root=root, split="train"), selected_indices)
    val_dataset = filter_labels(PPI(root=root, split="val"), selected_indices)
    test_dataset = filter_labels(PPI(root=root, split="test"), selected_indices)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    in_dim = train_dataset[0].x.shape[1]
    num_labels = train_dataset[0].y.shape[1]
    hidden_dim = 256
    embed_dim = 256

    encoder = GraphSAGEEncoder(in_dim, hidden_dim, embed_dim).to(device)
    head = LaMPHead(embed_dim, num_labels, label_dim=128, n_heads=4, n_rounds=2).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=0.005
    )

    best_val_micro_f1 = -1.0
    best_test_metrics = None
    epochs = 100

    for epoch in range(1, epochs + 1):
        loss = train_epoch(encoder, head, train_loader, optimizer, device)
        val_metrics = evaluate(encoder, head, val_loader, device)
        if epoch == 1 or epoch == 5:
            # One-time diagnostic: are logits actually varying, or collapsed?
            encoder.eval()
            head.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_loader)).to(device)
                z = encoder(sample_batch.x, sample_batch.edge_index)
                logits = head(z)
                print(f"  [diagnostic] logits: mean={logits.mean().item():.4f}, "
                      f"std={logits.std().item():.4f}, min={logits.min().item():.4f}, "
                      f"max={logits.max().item():.4f}")
                print(f"  [diagnostic] std across labels (per instance, averaged): "
                      f"{logits.std(dim=1).mean().item():.4f}")
                print(f"  [diagnostic] std across instances (per label, averaged): "
                      f"{logits.std(dim=0).mean().item():.4f}")
        if val_metrics["micro_f1"] > best_val_micro_f1:
            best_val_micro_f1 = val_metrics["micro_f1"]
            best_test_metrics = evaluate(encoder, head, test_loader, device)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss {loss:.4f} | "
                  f"Val micro-F1 {val_metrics['micro_f1']:.4f} | "
                  f"Best test micro-F1 {best_test_metrics['micro_f1']:.4f}")

    print(f"\nFinal test metrics (LaMP, GraphSAGE instance encoder):")
    print(format_metrics(best_test_metrics))

    os.makedirs("results", exist_ok=True)
    with open("results/lamp_style_baseline_results.json", "w") as f:
        json.dump(
            {
                "method": "GraphSAGE + LaMP (attention-based label message passing, 2 rounds, structure-agnostic)",
                "n_rounds": 2,
                "best_val_micro_f1": best_val_micro_f1,
                "test_metrics": best_test_metrics,
                "epochs": epochs,
            },
            f,
            indent=2,
        )
    print("Saved results to results/lamp_style_baseline_results.json")


if __name__ == "__main__":
    main()