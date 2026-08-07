"""
Runs plain independent-classifier baseline (GCN encoder, matching paper's
own setup) on Humloc, in BOTH transductive and inductive-ized protocols.

Validates against paper's own published GCN numbers (Table 2, Humloc):
  ranking_loss=13.29, hamming_loss=7.52, macro_auc=69.10, micro_auc=85.39,
  macro_ap=24.65, micro_ap=46.46, lrap=64.66

Usage:
    python src/validate_baseline_humloc.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(__file__))
from data_loaders import load_humloc, make_inductive_split
from train_corgcn_precise import GCNEncoder
from metrics import compute_full_metrics


class IndependentHead(nn.Module):
    def __init__(self, in_dim, num_labels):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_labels)

    def forward(self, z):
        return self.linear(z)


def train_transductive(data, train_mask, val_mask, test_mask, device, epochs=200, hidden_dim=64):
    in_dim = data.x.shape[1]
    num_labels = data.y.shape[1]
    encoder = GCNEncoder(in_dim, hidden_dim, hidden_dim).to(device)
    head = IndependentHead(hidden_dim, num_labels).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=0.001)

    x, edge_index, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)
    best_val, best_test = -1.0, None
    for epoch in range(1, epochs + 1):
        encoder.train(); head.train()
        optimizer.zero_grad()
        logits = head(encoder(x, edge_index))
        loss = F.binary_cross_entropy_with_logits(logits[train_mask], y[train_mask])
        loss.backward(); optimizer.step()

        with torch.no_grad():
            encoder.eval(); head.eval()
            logits_eval = head(encoder(x, edge_index))
            val_probs = torch.sigmoid(logits_eval[val_mask]).cpu().numpy()
            val_metrics = compute_full_metrics(y[val_mask].cpu().numpy(), val_probs)
        if val_metrics["micro_auc"] > best_val:
            best_val = val_metrics["micro_auc"]
            test_probs = torch.sigmoid(logits_eval[test_mask]).cpu().numpy()
            best_test = compute_full_metrics(y[test_mask].cpu().numpy(), test_probs)
    return best_test


def train_inductive(train_data, val_data, test_data, device, epochs=200, hidden_dim=64):
    in_dim = train_data.x.shape[1]
    num_labels = train_data.y.shape[1]
    encoder = GCNEncoder(in_dim, hidden_dim, hidden_dim).to(device)
    head = IndependentHead(hidden_dim, num_labels).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=0.001)

    tx, tei, ty = train_data.x.to(device), train_data.edge_index.to(device), train_data.y.to(device)
    vx, vei, vy = val_data.x.to(device), val_data.edge_index.to(device), val_data.y.to(device)
    ttx, ttei, tty = test_data.x.to(device), test_data.edge_index.to(device), test_data.y.to(device)

    best_val, best_test = -1.0, None
    for epoch in range(1, epochs + 1):
        encoder.train(); head.train()
        optimizer.zero_grad()
        logits = head(encoder(tx, tei))
        loss = F.binary_cross_entropy_with_logits(logits, ty)
        loss.backward(); optimizer.step()

        with torch.no_grad():
            encoder.eval(); head.eval()
            val_logits = head(encoder(vx, vei))
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_metrics = compute_full_metrics(vy.cpu().numpy(), val_probs)
            if val_metrics["micro_auc"] > best_val:
                best_val = val_metrics["micro_auc"]
                test_logits = head(encoder(ttx, ttei))
                test_probs = torch.sigmoid(test_logits).cpu().numpy()
                best_test = compute_full_metrics(tty.cpu().numpy(), test_probs)
    return best_test


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nLoading Humloc (transductive)...")
    data, train_mask, val_mask, test_mask = load_humloc()

    print("\n=== TRANSDUCTIVE baseline (GCN encoder) ===")
    trans_test = train_transductive(data, train_mask, val_mask, test_mask, device)
    for k, v in trans_test.items():
        print(f"  {k}: {v*100:.2f}")

    print("\nConstructing inductive-ized split...")
    train_data, val_data, test_data = make_inductive_split(data)

    print("\n=== INDUCTIVE-IZED baseline (GCN encoder) ===")
    induct_test = train_inductive(train_data, val_data, test_data, device)
    for k, v in induct_test.items():
        print(f"  {k}: {v*100:.2f}")

    print("\n=== PAPER'S PUBLISHED GCN (Table 2, Humloc, transductive) ===")
    print("  ranking_loss: 13.29, hamming_loss: 7.52, macro_auc: 69.10, "
          "micro_auc: 85.39, macro_ap: 24.65, micro_ap: 46.46, lrap: 64.66")

    print("\n=== COMPARISON ===")
    print(f"  Micro-AUC — Paper: 85.39, Ours (transductive): {trans_test['micro_auc']*100:.2f}, "
          f"Ours (inductive-ized): {induct_test['micro_auc']*100:.2f}")


if __name__ == "__main__":
    main()
