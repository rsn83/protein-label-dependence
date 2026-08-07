"""
Baseline + learnable pairwise label-dependency correction.

Controlled experiment: same GraphSAGE encoder, same discriminative training
via backprop, but adds a lightweight correction term using the SAME Bayesian
network structure learned earlier (data/label_structure.json) — reusing the
generative structure-learning result inside a discriminative model.

final_logits[c] = independent_logits[c] + sum_{p in parents(c)} w_{p,c} * sigmoid(independent_logits[p])

Only one learnable scalar per structure-learned edge (150 total) — lightweight,
directly comparable to the plain independent-classifier baseline.

Usage:
    python src/train_dependency_baseline.py
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
import json
import os
import sys

sys.path.append(os.path.dirname(__file__))
from train_baseline import GraphSAGEEncoder, load_selected_labels, filter_labels
from metrics import compute_all_metrics, format_metrics


class DependencyCorrectionHead(torch.nn.Module):
    def __init__(self, in_dim, num_labels, edges):
        """edges: list of (parent_idx, child_idx) tuples from learned BN structure."""
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, num_labels)
        self.num_labels = num_labels
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        self.register_buffer("src_idx", torch.tensor(src, dtype=torch.long))
        self.register_buffer("dst_idx", torch.tensor(dst, dtype=torch.long))
        self.edge_weights = torch.nn.Parameter(torch.zeros(len(edges)))

    def forward(self, z):
        independent_logits = self.linear(z)
        if len(self.src_idx) == 0:
            return independent_logits
        parent_probs = torch.sigmoid(independent_logits)[:, self.src_idx]  # (N, num_edges)
        weighted = parent_probs * self.edge_weights.unsqueeze(0)  # (N, num_edges)
        correction = torch.zeros_like(independent_logits)
        correction.index_add_(1, self.dst_idx, weighted)
        return independent_logits + correction


def load_structure_edges():
    with open("data/label_structure.json") as f:
        structure = json.load(f)
    parents = {int(k): v for k, v in structure["parents"].items()}
    edges = []
    for child, pa_list in parents.items():
        for parent in pa_list:
            edges.append((parent, child))
    return edges


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

    edges = load_structure_edges()
    print(f"Using {len(edges)} structure-learned edges for dependency correction.")

    encoder = GraphSAGEEncoder(in_dim, hidden_dim, embed_dim).to(device)
    head = DependencyCorrectionHead(embed_dim, num_labels, edges).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=0.005
    )

    best_val_micro_f1 = 0
    best_test_metrics = None
    epochs = 100

    for epoch in range(1, epochs + 1):
        loss = train_epoch(encoder, head, train_loader, optimizer, device)
        val_metrics = evaluate(encoder, head, val_loader, device)
        if val_metrics["micro_f1"] > best_val_micro_f1:
            best_val_micro_f1 = val_metrics["micro_f1"]
            best_test_metrics = evaluate(encoder, head, test_loader, device)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss {loss:.4f} | "
                  f"Val micro-F1 {val_metrics['micro_f1']:.4f} | "
                  f"Best test micro-F1 {best_test_metrics['micro_f1']:.4f}")

    print(f"\nFinal test metrics (dependency-correction baseline):")
    print(format_metrics(best_test_metrics))

    os.makedirs("results", exist_ok=True)
    with open("results/dependency_baseline_results.json", "w") as f:
        json.dump(
            {
                "method": "GraphSAGE + independent classifier + structure-constrained dependency correction",
                "n_edges_used": len(edges),
                "best_val_micro_f1": best_val_micro_f1,
                "test_metrics": best_test_metrics,
                "epochs": epochs,
            },
            f,
            indent=2,
        )
    print("Saved results to results/dependency_baseline_results.json")


if __name__ == "__main__":
    main()