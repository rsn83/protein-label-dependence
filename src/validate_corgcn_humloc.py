"""
CorGCN (Bei et al., KDD 2025), faithfully implemented against paper Eq 7-18,
adapted to sit on top of GraphSAGE embeddings (z) instead of raw features.

Simplifications (stated, not hidden):
  1. Macro label prototypes (K'=20), matching the paper's OWN large-label-space
     extension (Sec 4.4) used for their 121-label PPI experiment — but learned
     directly end-to-end rather than via their two-stage pretrain+cluster.
  2. Auxiliary losses L_cmi, L_lm (Eq 4-6) ARE included, matching the authors'
     real code (train_corgcn_precise.py was corrected after reviewing their
     actual GitHub implementation). Macro-label targets for these losses use
     a fixed grouping (see build_macro_group_map), not their learned
     label2cluster k-means procedure.
  3. This is a REPLACE-style method (no independent-classifier residual),
     matching the paper exactly — a real test of whether today's "replace vs
     augment" finding generalizes to a more sophisticated architecture.

Usage:
    python src/train_corgcn_precise.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import numpy as np
import json
import os
import sys

sys.path.append(os.path.dirname(__file__))
from train_baseline import GraphSAGEEncoder, load_selected_labels, filter_labels
from metrics import compute_all_metrics, format_metrics


class GCNEncoder(nn.Module):
    """Transductive-designed, matches CorGCN paper's default backbone."""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


class MultiLabelFocalLoss(nn.Module):
    """Matches the authors' real implementation exactly."""
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def dense_adj_from_edge_index(edge_index, n):
    A = torch.zeros(n, n, device=edge_index.device)
    A[edge_index[0], edge_index[1]] = 1.0
    return A


def build_macro_group_map(num_labels, k_prime, device):
    """
    Fixed, deterministic grouping of true labels into k_prime macro groups —
    a stated simplification of the authors' label2cluster (k-means on
    pretrained label embeddings), which would require an additional
    pretraining stage not otherwise needed here.
    """
    group_size = num_labels // k_prime
    groups = torch.arange(num_labels, device=device) // max(group_size, 1)
    groups = groups.clamp(max=k_prime - 1)
    return groups  # (num_labels,) -> which macro group each true label belongs to


def true_to_macro_targets(y, group_map, k_prime):
    """y: (n, num_labels) -> macro_y: (n, k_prime), 1 if ANY true label in that group is active."""
    n = y.shape[0]
    macro_y = torch.zeros(n, k_prime, device=y.device)
    macro_y.scatter_add_(1, group_map.unsqueeze(0).expand(n, -1), y.float())
    return (macro_y > 0).float()


def compute_focal_alpha(train_loader, group_map, k_prime, device):
    """Matches authors' focal_loss_init — inverse-sqrt class frequency weighting."""
    total = torch.zeros(k_prime, device=device)
    for batch in train_loader:
        macro_y = true_to_macro_targets(batch.y.to(device), group_map, k_prime)
        total += macro_y.sum(dim=0)
    class_weights = 1.0 / torch.sqrt(total + 1e-9)
    alpha = class_weights / class_weights.sum()
    return alpha


def gcn_normalize(A):
    A_hat = A + torch.eye(A.shape[0], device=A.device)
    deg = A_hat.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
    D = torch.diag(deg_inv_sqrt)
    return D @ A_hat @ D


class CorGCNHead(nn.Module):
    def __init__(self, embed_dim, num_labels, k_prime=20, proto_dim=64, top_lambda=5, dropout=0.3):
        super().__init__()
        self.num_labels = num_labels
        self.k_prime = k_prime
        self.proto_dim = proto_dim
        self.top_lambda = top_lambda

        # Real label embedding module (matches their label_emb + label_encoder)
        self.label_emb_table = nn.Embedding(k_prime, proto_dim)
        nn.init.xavier_uniform_(self.label_emb_table.weight.data)
        self.label_encoder = nn.Linear(proto_dim, proto_dim)

        self.feature_transform = nn.Linear(embed_dim, proto_dim)

        self.gcn_weight = nn.Linear(proto_dim, proto_dim)
        self.gcn_weight_view0 = nn.Linear(proto_dim, proto_dim)

        self.W1 = nn.Linear(proto_dim, proto_dim)
        self.W2 = nn.Linear(proto_dim, proto_dim)
        self.W3 = nn.Linear(proto_dim, proto_dim)

        self.classifier = nn.Linear(proto_dim * 2, num_labels)

        # Decoder for L_lm (matches their self.decoder) — maps proto_dim -> k_prime
        # (macro-label space), NOT num_labels — lm_loss compares against macro_targets
        self.decoder = nn.Sequential(
            nn.Linear(proto_dim, proto_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proto_dim, k_prime),
        )

    def get_label_prototypes(self):
        idx = torch.arange(self.k_prime, device=self.label_emb_table.weight.device)
        return F.normalize(self.label_encoder(self.label_emb_table(idx)), dim=-1)

    def cmi_loss(self, feat_emb, macro_targets, temp=1.0):
        """Matches authors' cmi_loss exactly — InfoNCE over label prototypes,
        positive mask = the instance's own (macro) label vector."""
        label_prototypes = self.get_label_prototypes()
        anchor_dot_contrast = torch.div(feat_emb @ label_prototypes.T, temp)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mask = macro_targets.float()
        denom = mask.sum(1).clamp(min=1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / denom
        return -mean_log_prob_pos.mean()

    def lm_loss(self, feat_emb, macro_targets, focal_loss_fn):
        """Matches authors' lm_loss exactly."""
        label_prototypes = self.get_label_prototypes()
        mlp_x = self.decoder(feat_emb)
        denom = macro_targets.float().sum(1, keepdim=True).clamp(min=1)
        z_label = (macro_targets.float() @ label_prototypes) / denom
        mlp_l = self.decoder(z_label)
        loss1 = focal_loss_fn(mlp_x, macro_targets.float())
        loss2 = focal_loss_fn(mlp_l, macro_targets.float())
        return (loss1 + loss2) / 2

    def forward(self, z, edge_index):
        n = z.shape[0]
        device = z.device

        E_x = self.feature_transform(z)
        label_prototypes = self.get_label_prototypes()

        E_x_norm = F.normalize(E_x, dim=-1)
        # (cos + 1) / 2 rescaling — matches authors' code exactly (not raw cosine)
        w = (E_x_norm @ label_prototypes.T + 1) / 2
        E_proj = w.unsqueeze(-1) * E_x.unsqueeze(1)

        A0 = dense_adj_from_edge_index(edge_index, n)
        A0_hat = A0 + torch.eye(n, device=device)
        deg0 = A0_hat.sum(dim=1, keepdim=True).clamp(min=1)
        E_sd = torch.einsum('ij,jkd->ikd', A0_hat, E_proj) / deg0.unsqueeze(-1)

        Z_hat_views = []
        for k in range(self.k_prime):
            v = F.normalize(E_sd[:, k, :], dim=-1)
            S_k = v @ v.T
            lam = min(self.top_lambda, n - 1)
            topk_vals, topk_idx = torch.topk(S_k, lam + 1, dim=1)
            A_k = torch.zeros(n, n, device=device)
            A_k.scatter_(1, topk_idx, 1.0)
            A_k_norm = gcn_normalize(A_k)

            z_k = F.leaky_relu(self.gcn_weight(A_k_norm @ E_proj[:, k, :]))
            Z_hat_views.append(z_k)
        Z_hat = torch.stack(Z_hat_views, dim=1)

        A0_norm = gcn_normalize(A0)
        Z0 = F.leaky_relu(self.gcn_weight_view0(A0_norm @ E_x))

        Q = self.W1(label_prototypes)
        Kmat = self.W2(Z_hat)
        attn_logits = torch.einsum('kd,nkd->nk', Q, Kmat) / (self.proto_dim ** 0.5)
        Cor = F.softmax(attn_logits, dim=-1)
        Z_views = self.W3(Cor.unsqueeze(-1) * Z_hat)

        # (cos + 1) / 2 rescaling here too, matching authors' final aggregation
        view_norm = F.normalize(Z_views, dim=-1)
        sim_weights = (torch.einsum('nkd,kd->nk', view_norm, label_prototypes) + 1) / 2
        sim_weights = sim_weights / sim_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weighted_views = (sim_weights.unsqueeze(-1) * Z_views).sum(dim=1)
        Z_cls = torch.cat([weighted_views, Z0], dim=-1)

        logits = self.classifier(Z_cls)
        return logits, E_x  # return E_x too, needed for cmi_loss/lm_loss during training


def train_epoch(encoder, head, loader, optimizer, device, group_map, k_prime, focal_loss_fn):
    encoder.train()
    head.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        z = encoder(batch.x, batch.edge_index)
        logits, E_x = head(z, batch.edge_index)

        macro_y = true_to_macro_targets(batch.y, group_map, k_prime)
        L_cls = F.binary_cross_entropy_with_logits(logits, batch.y)
        L_cmi = head.cmi_loss(E_x, macro_y)
        L_lm = head.lm_loss(E_x, macro_y, focal_loss_fn)

        # Adaptive weighting per Eq 20 (paper): alpha = |L_cls / 3*L_cmi|, similarly for beta
        alpha = (L_cls.detach() / (3 * L_cmi.detach() + 1e-8)).abs()
        beta = (L_cls.detach() / (3 * L_lm.detach() + 1e-8)).abs()
        loss = L_cls + alpha * L_cmi + beta * L_lm

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
        logits, _ = head(z, batch.edge_index)
        probs.append(torch.sigmoid(logits).cpu())
        ys.append(batch.y.cpu())
    probs = torch.cat(probs, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()
    return compute_all_metrics(ys, probs)


def run_full(encoder_type, device, epochs=100):
    root = "./data/raw"
    selected_indices = load_selected_labels()
    train_dataset = filter_labels(PPI(root=root, split="train"), selected_indices)
    val_dataset = filter_labels(PPI(root=root, split="val"), selected_indices)
    test_dataset = filter_labels(PPI(root=root, split="test"), selected_indices)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    in_dim = train_dataset[0].x.shape[1]
    num_labels = train_dataset[0].y.shape[1]
    hidden_dim = 256
    embed_dim = 256

    if encoder_type == "sage":
        encoder = GraphSAGEEncoder(in_dim, hidden_dim, embed_dim).to(device)
    elif encoder_type == "gcn":
        encoder = GCNEncoder(in_dim, hidden_dim, embed_dim).to(device)
    else:
        raise ValueError(encoder_type)

    head = CorGCNHead(embed_dim, num_labels, k_prime=20, proto_dim=64, top_lambda=5).to(device)

    group_map = build_macro_group_map(num_labels, head.k_prime, device)
    print(f"[{encoder_type}] Computing focal loss alpha weights...")
    focal_alpha = compute_focal_alpha(train_loader, group_map, head.k_prime, device)
    focal_loss_fn = MultiLabelFocalLoss(alpha=focal_alpha)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=0.005
    )

    best_val_micro_f1 = -1.0
    best_test_metrics = None
    epochs = 100

    for epoch in range(1, epochs + 1):
        loss = train_epoch(encoder, head, train_loader, optimizer, device, group_map, head.k_prime, focal_loss_fn)
        val_metrics = evaluate(encoder, head, val_loader, device)
        if val_metrics["micro_f1"] > best_val_micro_f1:
            best_val_micro_f1 = val_metrics["micro_f1"]
            best_test_metrics = evaluate(encoder, head, test_loader, device)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{encoder_type}] Epoch {epoch:03d} | Loss {loss:.4f} | "
                  f"Val micro-F1 {val_metrics['micro_f1']:.4f} | "
                  f"Best test micro-F1 {best_test_metrics['micro_f1']:.4f}")

    print(f"\n[{encoder_type}] Final test metrics:")
    print(format_metrics(best_test_metrics))
    return best_val_micro_f1, best_test_metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n=== SagePPI (true inductive) — GraphSAGE encoder ===")
    sage_val, sage_test = run_full("sage", device)

    print("\n=== SagePPI (true inductive) — GCN encoder ===")
    gcn_val, gcn_test = run_full("gcn", device)

    print("\n=== COMPARISON: GraphSAGE vs GCN on TRUE INDUCTIVE SagePPI ===")
    for k in sage_test:
        print(f"  {k}: GraphSAGE -> {sage_test[k]:.4f}, GCN -> {gcn_test[k]:.4f}, "
              f"delta -> {sage_test[k]-gcn_test[k]:+.4f}")

    print(f"\nCompare to Humloc (transductive) delta on micro_f1-equivalent (micro_auc): "
          f"GraphSAGE 87.28 vs GCN 86.25 -> delta +1.03")

    os.makedirs("results", exist_ok=True)
    with open("results/corgcn_encoder_comparison_results.json", "w") as f:
        json.dump(
            {
                "sage": {"val_micro_f1": sage_val, "test_metrics": sage_test},
                "gcn": {"val_micro_f1": gcn_val, "test_metrics": gcn_test},
            },
            f,
            indent=2,
        )
    print("Saved to results/corgcn_encoder_comparison_results.json")


if __name__ == "__main__":
    main()
