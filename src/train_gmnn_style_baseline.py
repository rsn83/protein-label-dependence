"""
GMNN-inspired baseline: label propagation over the PPI instance graph.

Unlike the earlier dependency-correction heads (which model label-label
correlation using the hill-climbing structure), this propagates predicted
LABEL PROBABILITIES between neighboring PROTEINS on the same PPI graph
GraphSAGE already uses — the idea being: a protein's predicted labels should
be informed by its interaction partners' predicted labels, independent of
any label-label structure.

Same GraphSAGEEncoder as every other variant. Only the head differs.

Usage:
    python src/train_gmnn_style_baseline.py
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
import json
import os
import sys

sys.path.append(os.path.dirname(__file__))
from train_baseline import GraphSAGEEncoder, load_selected_labels, filter_labels
from metrics import compute_all_metrics, format_metrics


class GMNNStyleHead(torch.nn.Module):
    """
    round 0: independent_logits = linear(z)
    round t: logits_t = independent_logits + propagate(sigmoid(logits_{t-1}), edge_index)
    'propagate' is a SAGEConv operating on label-probability vectors as if
    they were node features — same mechanism as GraphSAGE's own feature
    aggregation, but applied to predicted labels instead of raw input features.
    """

    def __init__(self, in_dim, num_labels, n_rounds=2):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, num_labels)
        self.n_rounds = n_rounds
        self.propagate_convs = torch.nn.ModuleList(
            [SAGEConv(num_labels, num_labels) for _ in range(n_rounds)]
        )

    def forward(self, z, edge_index):
        independent_logits = self.linear(z)
        logits = independent_logits
        for conv in self.propagate_convs:
            probs = torch.sigmoid(logits)
            propagated = conv(probs, edge_index)
            logits = independent_logits + propagated
        return logits


def train_epoch(encoder, head, loader, optimizer, device):
    encoder.train()
    head.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        z = encoder(batch.x, batch.edge_index)
        logits = head(z, batch.edge_index)
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
        logits = head(z, batch.edge_index)
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
    head = GMNNStyleHead(embed_dim, num_labels, n_rounds=2).to(device)

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

    print(f"\nFinal test metrics (GMNN-style instance-graph label propagation):")
    print(format_metrics(best_test_metrics))

    os.makedirs("results", exist_ok=True)
    with open("results/gmnn_style_baseline_results.json", "w") as f:
        json.dump(
            {
                "method": "GraphSAGE + independent classifier + GMNN-style label propagation over PPI graph (2 rounds)",
                "n_rounds": 2,
                "best_val_micro_f1": best_val_micro_f1,
                "test_metrics": best_test_metrics,
                "epochs": epochs,
            },
            f,
            indent=2,
        )
    print("Saved results to results/gmnn_style_baseline_results.json")


if __name__ == "__main__":
    main()