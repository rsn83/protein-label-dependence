"""
Precise GMNN (Qu, Bengio, Tang — ICML 2019), built directly against the
authors' own code (train.py, trainer.py, gnn.py from
github.com/DeepGraphLearning/GMNN, semisupervised/codes/).

Real mechanism (Algorithm 1 in the paper):
  - GNNq: standard 2-layer graph conv, INPUT = raw node features, predicts labels.
  - GNNp: standard 2-layer graph conv, INPUT = LABEL VECTORS (not features!),
    propagated over the same graph — this IS the label-dependency mechanism:
    a node's refined label is reconstructed from its NEIGHBORS' labels via
    the graph structure. Captures INTER-instance (neighbor-to-neighbor)
    dependency, NOT intra-instance (label-to-label) correlation.
  - Training alternates: pretrain q on ground truth -> repeat[ M-step: harden
    q's predictions, inject ground truth at labeled nodes, train p to
    reconstruct these label vectors through the graph -> E-step: use p's
    refined output (again with ground truth injected) as a soft distillation
    target, retrain q on raw features to match it ].
  - Final prediction uses q (paper: "qθ consistently outperforms pφ").

Multi-label adaptation (direct, stated): categorical softmax/cross-entropy
-> sigmoid/BCE throughout. Mechanism is otherwise unchanged from the paper.

This is a STANDALONE script — deliberately NOT folded into the generic
7-head sweep, since its two-phase alternating training doesn't fit the
single-forward-pass training loop every other method uses.

Usage:
    python src/gmnn_precise.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(__file__))
from data_loaders import load_cached_dataset
from metrics import compute_full_metrics, format_metrics


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, n):
        x = self.linear(x)
        src, dst = edge_index
        out = torch.zeros_like(x)
        out.index_add_(0, dst, x[src])
        deg = torch.zeros(n, device=x.device)
        deg.index_add_(0, dst, torch.ones(dst.shape[0], device=x.device))
        deg = deg.clamp(min=1).unsqueeze(-1)
        out = out / deg
        out = out + x
        return out


class GNNq(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_labels, dropout=0.5):
        super().__init__()
        self.m1 = GCNLayer(in_dim, hidden_dim)
        self.m2 = GCNLayer(hidden_dim, num_labels)
        self.dropout = dropout

    def forward(self, x, edge_index, n):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.m1(x, edge_index, n))
        x = F.dropout(x, self.dropout, training=self.training)
        return self.m2(x, edge_index, n)


class GNNp(nn.Module):
    def __init__(self, num_labels, hidden_dim, dropout=0.5):
        super().__init__()
        self.m1 = GCNLayer(num_labels, hidden_dim)
        self.m2 = GCNLayer(hidden_dim, num_labels)
        self.dropout = dropout

    def forward(self, label_vecs, edge_index, n):
        x = F.dropout(label_vecs, self.dropout, training=self.training)
        x = F.relu(self.m1(x, edge_index, n))
        x = F.dropout(x, self.dropout, training=self.training)
        return self.m2(x, edge_index, n)


def train_gmnn(x, edge_index, y, train_mask, val_mask, test_mask, device,
                hidden_dim=64, pre_epochs=200, epochs=50, n_iters=10, lr=0.01):
    n, in_dim = x.shape
    num_labels = y.shape[1]

    gnnq = GNNq(in_dim, hidden_dim, num_labels).to(device)
    gnnp = GNNp(num_labels, hidden_dim).to(device)
    opt_q = torch.optim.Adam(gnnq.parameters(), lr=lr, weight_decay=5e-4)
    opt_p = torch.optim.Adam(gnnp.parameters(), lr=lr, weight_decay=5e-4)

    def evaluate_q(mask):
        gnnq.eval()
        with torch.no_grad():
            logits = gnnq(x, edge_index, n)
            probs = torch.sigmoid(logits[mask]).cpu().numpy()
        return compute_full_metrics(y[mask].cpu().numpy(), probs)

    print("Pretraining GNNq...")
    best_val, best_test = -1.0, None
    for epoch in range(pre_epochs):
        gnnq.train()
        opt_q.zero_grad()
        logits = gnnq(x, edge_index, n)
        loss = F.binary_cross_entropy_with_logits(logits[train_mask], y[train_mask])
        loss.backward()
        opt_q.step()

        val_metrics = evaluate_q(val_mask)
        if val_metrics["micro_auc"] > best_val:
            best_val = val_metrics["micro_auc"]
            best_test = evaluate_q(test_mask)
    print(f"  Pretrain done. Best val micro_auc so far: {best_val:.4f}")

    for outer_iter in range(n_iters):
        gnnq.eval()
        with torch.no_grad():
            q_probs = torch.sigmoid(gnnq(x, edge_index, n))
        pseudo_labels = (q_probs > 0.5).float()
        pseudo_labels[train_mask] = y[train_mask]

        for epoch in range(epochs):
            gnnp.train()
            opt_p.zero_grad()
            logits_p = gnnp(pseudo_labels, edge_index, n)
            loss = F.binary_cross_entropy_with_logits(logits_p, pseudo_labels)
            loss.backward()
            opt_p.step()

        gnnp.eval()
        with torch.no_grad():
            p_probs = torch.sigmoid(gnnp(pseudo_labels, edge_index, n))
        soft_targets = p_probs.clone()
        soft_targets[train_mask] = y[train_mask]

        for epoch in range(epochs):
            gnnq.train()
            opt_q.zero_grad()
            logits_q = gnnq(x, edge_index, n)
            loss = F.binary_cross_entropy_with_logits(logits_q, soft_targets)
            loss.backward()
            opt_q.step()

            val_metrics = evaluate_q(val_mask)
            if val_metrics["micro_auc"] > best_val:
                best_val = val_metrics["micro_auc"]
                best_test = evaluate_q(test_mask)

        print(f"Outer iter {outer_iter + 1}/{n_iters}: best val micro_auc so far = {best_val:.4f}")

    return best_test


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data, train_mask, val_mask, test_mask = load_cached_dataset("humloc", "transductive", trial=0)
    x, edge_index, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)

    result = train_gmnn(x, edge_index, y, train_mask, val_mask, test_mask, device)

    print("\n=== Precise GMNN final test metrics (Humloc, transductive) ===")
    print(format_metrics(result))


if __name__ == "__main__":
    main()
