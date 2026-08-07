"""
Unified loaders for ALL 5 fixed datasets: Humloc, PCG, EukaryoteGo,
Blogcatalog, SagePPI.

Every loader returns a consistent interface for the TRANSDUCTIVE protocol:
    data: torch_geometric.data.Data (x, edge_index, y)
    train_mask, val_mask, test_mask: (n,) bool tensors

For the INDUCTIVE protocol:
    - CSV/mat datasets (Humloc, PCG, EukaryoteGo, Blogcatalog): three
      separate Data objects (train_data, val_data, test_data) built by
      partitioning the transductive graph and discarding cross-partition
      edges (make_inductive_split).
    - SagePPI: uses its NATIVE inductive structure (24 separate real
      graphs) — not a construction, the actual original dataset design.
"""

import torch
import pandas as pd
import numpy as np
import os
from torch_geometric.data import Data
from torch_geometric.datasets import PPI
from scipy.io import loadmat
from sklearn.decomposition import TruncatedSVD


def make_random_masks(n, seed=0, train_frac=0.6, val_frac=0.2):
    torch.manual_seed(seed)
    perm = torch.randperm(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True
    return train_mask, val_mask, test_mask


def make_inductive_split(data, seed=0, train_frac=0.6, val_frac=0.2):
    n = data.x.shape[0]
    train_mask, val_mask, test_mask = make_random_masks(n, seed, train_frac, val_frac)

    def build_subgraph(mask):
        node_ids = mask.nonzero(as_tuple=True)[0]
        remap = torch.full((n,), -1, dtype=torch.long)
        remap[node_ids] = torch.arange(len(node_ids))

        src, dst = data.edge_index
        edge_in_partition = mask[src] & mask[dst]
        local_src = remap[src[edge_in_partition]]
        local_dst = remap[dst[edge_in_partition]]
        local_edge_index = torch.stack([local_src, local_dst])

        return Data(x=data.x[node_ids], edge_index=local_edge_index, y=data.y[node_ids])

    train_data = build_subgraph(train_mask)
    val_data = build_subgraph(val_mask)
    test_data = build_subgraph(test_mask)

    print(f"  Inductive-ized: train {train_data.x.shape[0]}n/{train_data.edge_index.shape[1]}e, "
          f"val {val_data.x.shape[0]}n/{val_data.edge_index.shape[1]}e, "
          f"test {test_data.x.shape[0]}n/{test_data.edge_index.shape[1]}e")

    return train_data, val_data, test_data


def _load_csv_dataset(data_dir, edge_file, split_file="split.pt"):
    features = pd.read_csv(f"{data_dir}/features.csv", header=None).values.astype(np.float32)
    labels = pd.read_csv(f"{data_dir}/labels.csv", header=None).values.astype(np.float32)
    edges = pd.read_csv(f"{data_dir}/{edge_file}")

    n = features.shape[0]
    cols = edges.columns.tolist()
    if "prot1" in cols and "prot2" in cols:
        src = edges["prot1"].values.astype(np.int64)
        dst = edges["prot2"].values.astype(np.int64)
    else:
        src = edges.iloc[:, 0].values.astype(np.int64)
        dst = edges.iloc[:, 1].values.astype(np.int64)

    edge_index = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)

    split_path = f"{data_dir}/{split_file}"
    if os.path.exists(split_path):
        split = torch.load(split_path, weights_only=False)
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)
        train_mask[split["train_mask"]] = True
        val_mask[split["val_mask"]] = True
        test_mask[split["test_mask"]] = True
        print(f"  Using authors' own split from {split_file}")
    else:
        train_mask, val_mask, test_mask = make_random_masks(n)
        print(f"  No split.pt found — generated random 60/20/20 split (seed=0)")

    print(f"  {n} nodes, {y.shape[1]} labels, {x.shape[1]} features, "
          f"{train_mask.sum().item()}/{val_mask.sum().item()}/{test_mask.sum().item()} train/val/test")

    return Data(x=x, edge_index=edge_index, y=y), train_mask, val_mask, test_mask


def load_humloc(data_dir="MLGNC/data/HumanGo"):
    print("Loading Humloc...")
    return _load_csv_dataset(data_dir, edge_file="edge_list.csv")


def load_pcg(data_dir="MLGNC/data/pcg_removed_isolated_nodes"):
    print("Loading PCG...")
    return _load_csv_dataset(data_dir, edge_file="edges_undir.csv")


def load_eukaryotego(data_dir="MLGNC/data/EukaryoteGo"):
    print("Loading EukaryoteGo...")
    return _load_csv_dataset(data_dir, edge_file="edge_list.csv")


def load_blogcatalog(mat_path="MLGNC/data/blogcatalog.mat",
                      split_dir="MLGNC/data/blogcatalog_0.6", feature_dim=32, seed=0):
    print("Loading Blogcatalog...")
    mat = loadmat(mat_path)
    network = mat["network"].tocoo()
    group = mat["group"].tocoo()

    n = network.shape[0]
    y = torch.tensor(group.toarray(), dtype=torch.float32)

    src = torch.tensor(network.row, dtype=torch.long)
    dst = torch.tensor(network.col, dtype=torch.long)
    edge_index = torch.stack([src, dst])

    print(f"  No node features in source data — computing {feature_dim}-dim "
          f"structural embedding via SVD of adjacency matrix...")
    svd = TruncatedSVD(n_components=feature_dim, random_state=seed)
    x_np = svd.fit_transform(mat["network"])
    x = torch.tensor(x_np, dtype=torch.float32)

    train_mask = val_mask = test_mask = None
    if os.path.isdir(split_dir):
        candidates = sorted([f for f in os.listdir(split_dir) if f.endswith(".pt")])
        if candidates:
            split_path = f"{split_dir}/{candidates[0]}"
            split = torch.load(split_path, weights_only=False)
            train_mask = torch.zeros(n, dtype=torch.bool)
            val_mask = torch.zeros(n, dtype=torch.bool)
            test_mask = torch.zeros(n, dtype=torch.bool)
            train_mask[split["train_mask"]] = True
            val_mask[split["val_mask"]] = True
            test_mask[split["test_mask"]] = True
            print(f"  Using authors' own split from {split_path}")
    if train_mask is None:
        train_mask, val_mask, test_mask = make_random_masks(n, seed)
        print(f"  Generated random 60/20/20 split (seed={seed})")

    print(f"  {n} nodes, {y.shape[1]} labels, {x.shape[1]} features (SVD-derived), "
          f"{train_mask.sum().item()}/{val_mask.sum().item()}/{test_mask.sum().item()} train/val/test")

    return Data(x=x, edge_index=edge_index, y=y), train_mask, val_mask, test_mask


def load_sageppi_native_inductive(selected_indices, root="./data/raw"):
    from train_baseline import filter_labels
    print("Loading SagePPI (native inductive)...")
    train_dataset = filter_labels(PPI(root=root, split="train"), selected_indices)
    val_dataset = filter_labels(PPI(root=root, split="val"), selected_indices)
    test_dataset = filter_labels(PPI(root=root, split="test"), selected_indices)
    print(f"  {len(train_dataset)} train graphs, {len(val_dataset)} val, {len(test_dataset)} test")
    return train_dataset, val_dataset, test_dataset


def load_sageppi_transductive(selected_indices, root="./data/raw", seed=0):
    from train_baseline import filter_labels
    print("Loading SagePPI (transductive, merged)...")
    all_graphs = (
        list(filter_labels(PPI(root=root, split="train"), selected_indices)) +
        list(filter_labels(PPI(root=root, split="val"), selected_indices)) +
        list(filter_labels(PPI(root=root, split="test"), selected_indices))
    )

    xs, ys, edge_indices = [], [], []
    node_offset = 0
    for g in all_graphs:
        xs.append(g.x)
        ys.append(g.y)
        edge_indices.append(g.edge_index + node_offset)
        node_offset += g.x.shape[0]

    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    edge_index = torch.cat(edge_indices, dim=1)
    n = x.shape[0]
    print(f"  Merged {len(all_graphs)} graphs: {n} nodes, {edge_index.shape[1]} edges")

    train_mask, val_mask, test_mask = make_random_masks(n, seed)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data, train_mask, val_mask, test_mask


CSV_MAT_DATASET_LOADERS = {
    "humloc": load_humloc,
    "pcg": load_pcg,
    "eukaryotego": load_eukaryotego,
    "blogcatalog": load_blogcatalog,
}

ALL_DATASET_NAMES = list(CSV_MAT_DATASET_LOADERS.keys()) + ["sageppi"]


def cache_all_datasets(cache_dir="data/cached_datasets", selected_indices=None):
    os.makedirs(cache_dir, exist_ok=True)

    for name, loader_fn in CSV_MAT_DATASET_LOADERS.items():
        cache_path = f"{cache_dir}/{name}.pt"
        if os.path.exists(cache_path):
            print(f"{name}: already cached, skipping")
            continue
        print(f"\nBuilding cache for {name}...")
        data, train_mask, val_mask, test_mask = loader_fn()
        train_data, val_data, test_data = make_inductive_split(data)
        torch.save({
            "transductive": {"data": data, "train_mask": train_mask,
                              "val_mask": val_mask, "test_mask": test_mask},
            "inductive": {"train_data": train_data, "val_data": val_data, "test_data": test_data},
        }, cache_path)
        print(f"  Saved to {cache_path}")

    if selected_indices is not None:
        cache_path = f"{cache_dir}/sageppi.pt"
        if os.path.exists(cache_path):
            print("sageppi: already cached, skipping")
        else:
            print("\nBuilding cache for sageppi...")
            data, train_mask, val_mask, test_mask = load_sageppi_transductive(selected_indices)
            train_dataset, val_dataset, test_dataset = load_sageppi_native_inductive(selected_indices)
            torch.save({
                "transductive": {"data": data, "train_mask": train_mask,
                                  "val_mask": val_mask, "test_mask": test_mask},
                "inductive_native": {"train_dataset": train_dataset,
                                      "val_dataset": val_dataset, "test_dataset": test_dataset},
            }, cache_path)
            print(f"  Saved to {cache_path}")
    else:
        print("\nselected_indices not provided — skipping sageppi caching. "
              "Call cache_all_datasets(selected_indices=load_selected_labels()) to include it.")

    print(f"\nDone. Cached datasets in {cache_dir}/")


def load_cached_dataset(name, protocol, cache_dir="data/cached_datasets"):
    cache_path = f"{cache_dir}/{name}.pt"
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"{cache_path} not found — run cache_all_datasets() first")
    cached = torch.load(cache_path, weights_only=False)

    if protocol == "transductive":
        d = cached["transductive"]
        return d["data"], d["train_mask"], d["val_mask"], d["test_mask"]
    elif protocol == "inductive":
        if "inductive" in cached:
            d = cached["inductive"]
            return d["train_data"], d["val_data"], d["test_data"]
        elif "inductive_native" in cached:
            d = cached["inductive_native"]
            return d["train_dataset"], d["val_dataset"], d["test_dataset"]
        else:
            raise KeyError(f"No inductive data found for {name}")
    else:
        raise ValueError(protocol)


def load_selected_labels(path="data/selected_labels.json"):
    import json
    with open(path) as f:
        sel = json.load(f)
    return sel["selected_indices"]
