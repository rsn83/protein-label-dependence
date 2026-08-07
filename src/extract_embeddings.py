"""
Extract frozen GraphSAGE embeddings (Z) for train/val/test splits.

Loads the encoder checkpoint saved by train_baseline.py, runs it in eval
mode (no gradient updates) over all three splits, and saves the resulting
per-protein embeddings + labels + selected-label indices to disk. This Z
is the feature input to the ECAI 2016 mixture model — same role the
image-derived features played in the original paper.

Usage:
    python src/extract_embeddings.py
"""

import torch
import json
import os
import numpy as np
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
import sys

sys.path.append(os.path.dirname(__file__))
from train_baseline import GraphSAGEEncoder, load_selected_labels, filter_labels


@torch.no_grad()
def extract_split(encoder, dataset, device):
    encoder.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    all_z, all_y = [], []
    for batch in loader:
        batch = batch.to(device)
        z = encoder(batch.x, batch.edge_index)
        all_z.append(z.cpu().numpy())
        all_y.append(batch.y.cpu().numpy())
    return np.concatenate(all_z, axis=0), np.concatenate(all_y, axis=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("checkpoints/encoder_config.json") as f:
        config = json.load(f)

    encoder = GraphSAGEEncoder(
        config["in_dim"], config["hidden_dim"], config["embed_dim"]
    ).to(device)
    encoder.load_state_dict(torch.load("checkpoints/graphsage_encoder_best.pt", map_location=device))
    print("Loaded encoder checkpoint.")

    root = "./data/raw"
    selected_indices = load_selected_labels()

    train_dataset = filter_labels(PPI(root=root, split="train"), selected_indices)
    val_dataset = filter_labels(PPI(root=root, split="val"), selected_indices)
    test_dataset = filter_labels(PPI(root=root, split="test"), selected_indices)

    os.makedirs("data/embeddings", exist_ok=True)
    for name, dataset in [("train", train_dataset), ("val", val_dataset), ("test", test_dataset)]:
        Z, Y = extract_split(encoder, dataset, device)
        np.save(f"data/embeddings/{name}_Z.npy", Z)
        np.save(f"data/embeddings/{name}_Y.npy", Y)
        print(f"{name}: Z shape {Z.shape}, Y shape {Y.shape}")

    print("Saved frozen embeddings to data/embeddings/")


if __name__ == "__main__":
    main()