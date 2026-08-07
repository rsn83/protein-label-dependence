"""
Precise ML-GCN (Chen et al., CVPR 2019), adapted to use GraphSAGE as the
instance encoder instead of the original's CNN/ResNet.

Mechanism (matches the paper):
  1. Build a label co-occurrence matrix from training labels.
  2. Binarize with threshold tau, then reweight per the paper's scheme
     (Eq. 5-6): off-diagonal entries scaled by p / (row sum), diagonal = 1-p.
     This specific reweighting prevents over-smoothing in the GCN.
  3. Learnable label embeddings (substituting GloVe word embeddings — GO-terms
     don't have off-the-shelf semantic embeddings in this pipeline; this is
     the standard substitute when semantic embeddings are unavailable).
  4. 2-layer GCN over the FIXED (non-learnable) reweighted adjacency,
     producing a per-label classifier weight vector.
  5. Final logits = dot product of instance embedding (from GraphSAGE) and
     each label's GCN-produced classifier vector — NOT an additive correction
     like the earlier heads; this is ML-GCN's actual combination mechanism.

Usage:
    python src/train_mlgcn_style_baseline.py
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
import numpy as np
import json
import os
import sys

sys.path.append(os.path.dirname(__file__))
from train_baseline import GraphSAGEEncoder, load_selected_labels, filter_labels
from metrics import compute_all_metrics, format_metrics


def build_reweighted_adjacency(Y: np.ndarray, tau: float = 0.4, p: float = 0.2):
    """
    Y: (n_instances, n_labels) binary label matrix (training set).
    Returns: (n_labels, n_labels) reweighted adjacency, per ML-GCN Eq. 5-6.
    """
    n_labels = Y.shape[1]
    co_occur = Y.T @ Y  # (n_labels, n_labels), co_occur[i,j] = count both i and j are 1
    label_counts = Y.sum(axis=0)  # (n_labels,)
    cond_prob = co_occur / np.maximum(label_counts[:, None], 1)
    np.fill_diagonal(cond_prob, 0)

    binary_adj = (cond_prob >= tau).astype(float)

    row_sums = binary_adj.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-6)
    reweighted = p * binary_adj / row_sums
    np.fill_diagonal(reweighted, 1 - p)

    n_edges = (binary_adj.sum())
    print(f"Co-occurrence graph: tau={tau}, p={p}, {int(n_edges)} directed edges "
          f"after binarization (of {n_labels * (n_labels - 1)} possible off-diagonal entries)")
    return reweighted.astype(np.float32)


class MLGCNHead(torch.nn.Module):
    def __init__(self, instance_embed_dim, n_labels, adjacency: np.ndarray,
                 label_embed_dim: int = 128, gcn_hidden_dim: int = 256):
        super().__init__()
        self.register_buffer("adjacency", torch.tensor(adjacency))  # FIXED, not learnable
        self.label_embeddings = torch.nn.Parameter(torch.randn(n_labels, label_embed_dim) * 0.1)
        self.gcn1 = torch.nn.Linear(label_embed_dim, gcn_hidden_dim, bias=False)
        self.gcn2 = torch.nn.Linear(gcn_hidden_dim, instance_embed_dim, bias=False)

    def forward(self, instance_z):
        h = self.adjacency @ self.label_embeddings
        h = F.leaky_relu(self.gcn1(h), negative_slope=0.2)
        h = self.adjacency @ h
        classifier_vectors = F.leaky_relu(self.gcn2(h), negative_slope=0.2)
        logits = instance_z @ classifier_vectors.T
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

    Y_train = np.load("data/embeddings/train_Y.npy")
    adjacency = build_reweighted_adjacency(Y_train, tau=0.4, p=0.2)

    encoder = GraphSAGEEncoder(in_dim, hidden_dim, embed_dim).to(device)
    head = MLGCNHead(embed_dim, num_labels, adjacency).to(device)

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

    print(f"\nFinal test metrics (ML-GCN, GraphSAGE instance encoder):")
    print(format_metrics(best_test_metrics))

    os.makedirs("results", exist_ok=True)
    with open("results/mlgcn_style_baseline_results.json", "w") as f:
        json.dump(
            {
                "method": "GraphSAGE + ML-GCN (label co-occurrence GCN, learnable label embeddings substituting GloVe)",
                "tau": 0.4,
                "p": 0.2,
                "best_val_micro_f1": best_val_micro_f1,
                "test_metrics": best_test_metrics,
                "epochs": epochs,
            },
            f,
            indent=2,
        )
    print("Saved results to results/mlgcn_style_baseline_results.json")


if __name__ == "__main__":
    main()