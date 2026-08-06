"""
Download and inspect the PPI (SagePPI) dataset via PyTorch Geometric.

This is the multi-label protein-protein interaction benchmark from the
GraphSAGE paper (Hamilton et al., 2017), bundled directly in PyG.

Usage:
    python data/download.py
"""

from torch_geometric.datasets import PPI


def load_and_inspect(root: str = "./data/raw"):
    print("Loading PPI train split...")
    train_dataset = PPI(root=root, split="train")
    print("Loading PPI val split...")
    val_dataset = PPI(root=root, split="val")
    print("Loading PPI test split...")
    test_dataset = PPI(root=root, split="test")

    print(f"\nNumber of graphs — train: {len(train_dataset)}, "
          f"val: {len(val_dataset)}, test: {len(test_dataset)}")

    sample = train_dataset[0]
    print("\nFirst training graph:")
    print(sample)
    print(f"Node feature dim: {sample.x.shape[1]}")
    print(f"Num labels (multi-label targets): {sample.y.shape[1]}")
    print(f"Num nodes: {sample.num_nodes}, num edges: {sample.num_edges}")

    total_nodes = sum(g.num_nodes for g in train_dataset) + \
        sum(g.num_nodes for g in val_dataset) + \
        sum(g.num_nodes for g in test_dataset)
    total_edges = sum(g.num_edges for g in train_dataset) + \
        sum(g.num_edges for g in val_dataset) + \
        sum(g.num_edges for g in test_dataset)
    print(f"\nTotal nodes across all splits: {total_nodes}")
    print(f"Total edges across all splits: {total_edges}")

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    load_and_inspect()