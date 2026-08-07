"""
Discretize frozen GraphSAGE embeddings into bins.

The ECAI 2016 model requires discrete feature values (Multinomial component
distributions, Section 2.2 / Section 4.1 of the paper — original features
were discretized into 4 bins). GraphSAGE embeddings are continuous, so this
step is required before they can be used as input to the mixture model.

Bin edges are computed from the TRAINING set only, then applied identically
to val/test — standard practice, avoids leaking test-set distribution into
the discretization.

Usage:
    python src/discretize_embeddings.py --n_bins 4
"""

import argparse
import numpy as np
import os


def fit_bin_edges(Z_train: np.ndarray, n_bins: int):
    """Equal-frequency (quantile) binning per dimension, fit on train only."""
    n_dims = Z_train.shape[1]
    edges = np.zeros((n_dims, n_bins + 1))
    for d in range(n_dims):
        quantiles = np.linspace(0, 100, n_bins + 1)
        edges[d] = np.percentile(Z_train[:, d], quantiles)
        edges[d, 0] -= 1e-6   # ensure min value falls inside first bin
        edges[d, -1] += 1e-6  # ensure max value falls inside last bin
    return edges


def apply_bins(Z: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Returns integer bin index (0 to n_bins-1) per feature dimension."""
    n_dims = Z.shape[1]
    Z_disc = np.zeros_like(Z, dtype=int)
    for d in range(n_dims):
        Z_disc[:, d] = np.digitize(Z[:, d], edges[d][1:-1])  # interior edges only
    return Z_disc


def main(n_bins: int = 4):
    Z_train = np.load("data/embeddings/train_Z.npy")
    Z_val = np.load("data/embeddings/val_Z.npy")
    Z_test = np.load("data/embeddings/test_Z.npy")

    print(f"Fitting {n_bins}-bin quantile edges on training embeddings "
          f"(shape {Z_train.shape})...")
    edges = fit_bin_edges(Z_train, n_bins)

    for name, Z in [("train", Z_train), ("val", Z_val), ("test", Z_test)]:
        Z_disc = apply_bins(Z, edges)
        np.save(f"data/embeddings/{name}_Z_discrete.npy", Z_disc)
        print(f"{name}: discretized shape {Z_disc.shape}, "
              f"value range [{Z_disc.min()}, {Z_disc.max()}]")

    np.save("data/embeddings/bin_edges.npy", edges)
    print(f"Saved discretized embeddings and bin edges to data/embeddings/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_bins", type=int, default=4)
    args = parser.parse_args()
    main(n_bins=args.n_bins)