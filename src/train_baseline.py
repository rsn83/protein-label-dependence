"""
Baseline: GraphSAGE encoder + independent per-label classifier.

No label dependency modeling — each of the 121 labels is predicted
independently via a shared encoder + linear output layer with sigmoid.
This is the reference point your method (Stage 2 = 2016 mixture model)
gets compared against.

Usage:
    python src/train_baseline.py
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from sklearn.metrics import f1_score
import json
import os


class GraphSAGEEncoder(torch.nn.Module):
    """Stage 1 — shared across all methods in the comparison."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, out_dim))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x  # returns Z, the embedding — Stage 1 output


class IndependentLabelHead(torch.nn.Module):
    """Stage 2 — baseline only. No dependency modeling between labels."""

    def __init__(self, in_dim, num_labels):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, num_labels)

    def forward(self, z):
        return self.linear(z)  # raw logits, one per label, independent


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
    preds, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        z = encoder(batch.x, batch.edge_index)
        logits = head(z)
        preds.append((torch.sigmoid(logits) > 0.5).float().cpu())
        ys.append(batch.y.cpu())
    preds = torch.cat(preds, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()
    micro_f1 = f1_score(ys, preds, average="micro", zero_division=0)
    return micro_f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root = "./data/raw"
    train_dataset = PPI(root=root, split="train")
    val_dataset = PPI(root=root, split="val")
    test_dataset = PPI(root=root, split="test")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    in_dim = train_dataset.num_features   # 50
    num_labels = train_dataset[0].y.shape[1]  # 121
    hidden_dim = 256
    embed_dim = 256

    encoder = GraphSAGEEncoder(in_dim, hidden_dim, embed_dim).to(device)
    head = IndependentLabelHead(embed_dim, num_labels).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=0.005
    )

    best_val_f1 = 0
    best_test_f1 = 0
    epochs = 100

    for epoch in range(1, epochs + 1):
        loss = train_epoch(encoder, head, train_loader, optimizer, device)
        val_f1 = evaluate(encoder, head, val_loader, device)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_test_f1 = evaluate(encoder, head, test_loader, device)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss {loss:.4f} | "
                  f"Val micro-F1 {val_f1:.4f} | Best test micro-F1 {best_test_f1:.4f}")

    print(f"\nFinal: best val micro-F1 = {best_val_f1:.4f}, "
          f"corresponding test micro-F1 = {best_test_f1:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/baseline_results.json", "w") as f:
        json.dump(
            {
                "method": "GraphSAGE + independent classifier",
                "best_val_micro_f1": best_val_f1,
                "test_micro_f1": best_test_f1,
                "epochs": epochs,
            },
            f,
            indent=2,
        )
    print("Saved results to results/baseline_results.json")


if __name__ == "__main__":
    main()