"""
Timing test: Bayesian network structure learning on the 121-label matrix ALONE.

This answers the real feasibility question before any other code is written:
the ECAI 2016 paper reports ~20 hours runtime at just 14 labels, with runtime
stated to grow quadratically with the number of labels. This benchmark has
121 labels (~8.6x more). This test measures actual time on YOUR machine,
on the REAL label matrix, before committing further implementation time.

Usage:
    python src/test_structure_learning_scale.py
"""

import time
import numpy as np
import pandas as pd
from torch_geometric.datasets import PPI
from pgmpy.estimators import HillClimbSearch


def load_label_matrix():
    """Load just the label matrix Y from SagePPI train split — no features, no graph."""
    train_dataset = PPI(root="./data/raw", split="train")
    all_labels = []
    for graph in train_dataset:
        all_labels.append(graph.y.numpy())
    Y = np.concatenate(all_labels, axis=0)  # (num_instances, 121)
    return Y


def run_timing_test(max_labels_to_try=(10, 20, 40, 80, 121)):
    """
    Progressively test structure learning at increasing label counts,
    so if it becomes intractable, we know roughly where the wall is —
    not just a single all-or-nothing failure.
    """
    Y = load_label_matrix()
    print(f"Full label matrix shape: {Y.shape}")

    for n_labels in max_labels_to_try:
        if n_labels > Y.shape[1]:
            continue
        subset = Y[:, :n_labels].astype(int)
        col_names = [f"L{i}" for i in range(n_labels)]
        df = pd.DataFrame(subset, columns=col_names).astype("category")

        print(f"\n--- Testing structure learning on {n_labels} labels "
              f"({df.shape[0]} instances) ---")
        start = time.time()
        try:
            hc = HillClimbSearch(df)
            best_model = hc.estimate(
                scoring_method="bic-d",  # BIC for discrete data, current pgmpy API
                max_indegree=2,  # matches p=2 constraint from your ECAI paper
                max_iter=int(1e4),
            )
            elapsed = time.time() - start
            print(f"Completed in {elapsed:.1f} sec. "
                  f"Learned {len(best_model.edges())} edges.")
        except Exception as e:
            elapsed = time.time() - start
            print(f"FAILED after {elapsed:.1f} sec: {e}")
            print("Stopping escalation — this label count is not tractable as-is.")
            break


if __name__ == "__main__":
    run_timing_test()