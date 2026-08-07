"""
Ablation: MLP encoder (no graph, no neighbor aggregation) instead of GraphSAGE.

Tests Hypothesis 1 directly: intra-instance dependency mechanisms (like
dependency-correction) should show a LARGER relative improvement over baseline
when there's no redundant inter-instance signal already baked into Z by
GraphSAGE's neighbor aggregation. If the gain stays similarly small here too,
that would falsify the redundancy hypothesis.

Produces two results: MLP baseline, and MLP + dependency-correction (reusing
the same structure-learned edges from data/label_structure.json).

Usage:
    python src/train_mlp_ablation.py
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
import json
import os
import sys

sys.path.append(os.path.dirname(__file__))
from train_baseline import load_selected_labels, filter_labels
from train_dependency_baseline import DependencyCorrectionHead, load_structure_edges
from metrics import compute_all_metrics, format_metrics


class MLPEncoder(torch.nn.Module):
    """
    Deliberately ignores edge_index — no graph, no neighbor aggregation.
    Each protein's embedding depends ONLY on its own raw features. Same
    input/output dims as GraphSAGEEncoder, so it's a drop-in replacement.
    """

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index=None):  # edge_index accepted but unused
        return self.net(x)


class IndependentLabelHead(torch.nn.Module):
    def __init__(self, in_dim, num_labels):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, num_labels)

    def forward(self, z):
        return self.linear(z)


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


def run_variant(head_type, in_dim, num_labels, hidden_dim, embed_dim, edges,
                 train_loader, val_loader, test_loader, device, epochs=100):
    encoder = MLPEncoder(in_dim, hidden_dim, embed_dim).to(device)
    if head_type == "independent":
        head = IndependentLabelHead(embed_dim, num_labels).to(device)
    elif head_type == "dependency":
        head = DependencyCorrectionHead(embed_dim, num_labels, edges).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=0.005
    )

    best_val_micro_f1 = -1.0
    best_test_metrics = None

    for epoch in range(1, epochs + 1):
        loss = train_epoch(encoder, head, train_loader, optimizer, device)
        val_metrics = evaluate(encoder, head, val_loader, device)
        if val_metrics["micro_f1"] > best_val_micro_f1:
            best_val_micro_f1 = val_metrics["micro_f1"]
            best_test_metrics = evaluate(encoder, head, test_loader, device)
        if epoch % 20 == 0 or epoch == 1:
            print(f"  [{head_type}] Epoch {epoch:03d} | Loss {loss:.4f} | "
                  f"Val micro-F1 {val_metrics['micro_f1']:.4f} | "
                  f"Best test micro-F1 {best_test_metrics['micro_f1']:.4f}")

    return best_val_micro_f1, best_test_metrics


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
    print(f"Using {len(edges)} structure-learned edges for the dependency variant.\n")

    print("=== MLP baseline (no graph) ===")
    baseline_val, baseline_test = run_variant(
        "independent", in_dim, num_labels, hidden_dim, embed_dim, edges,
        train_loader, val_loader, test_loader, device
    )

    print("\n=== MLP + dependency correction (no graph) ===")
    dep_val, dep_test = run_variant(
        "dependency", in_dim, num_labels, hidden_dim, embed_dim, edges,
        train_loader, val_loader, test_loader, device
    )

    print("\n=== ABLATION RESULT ===")
    print(f"MLP baseline test micro-F1:              {baseline_test['micro_f1']:.4f}")
    print(f"MLP + dependency correction micro-F1:     {dep_test['micro_f1']:.4f}")
    delta = dep_test['micro_f1'] - baseline_test['micro_f1']
    print(f"Delta (dependency correction contribution): {delta:+.4f}")
    print(f"\nCompare to GraphSAGE encoder delta: +0.0070 (0.8815 -> 0.8885)")
    print(f"If this delta is LARGER, supports the redundancy hypothesis "
          f"(GraphSAGE's aggregation was absorbing some of the same signal).")

    os.makedirs("results", exist_ok=True)
    with open("results/mlp_ablation_results.json", "w") as f:
        json.dump(
            {
                "mlp_baseline": {"val_micro_f1": baseline_val, "test_metrics": baseline_test},
                "mlp_dependency_correction": {"val_micro_f1": dep_val, "test_metrics": dep_test},
                "delta_micro_f1": delta,
                "graphsage_delta_for_comparison": 0.0070,
            },
            f,
            indent=2,
        )
    print("\nSaved to results/mlp_ablation_results.json")


if __name__ == "__main__":
    main()