"""
Diagnostic: how many distinct label vectors occur in the training data,
and how many instances support each one?

The ECAI 2016 EM procedure estimates theta^l_{j,k} per DISTINCT FULL LABEL
VECTOR l (grouping instances by exact match). This was fine at the paper's
original scale (q=6 to 27 labels). At q=80, this check determines whether
that literal formulation is still statistically estimable, or whether it
needs a principled simplification before implementation.

Usage:
    python src/check_label_vector_cardinality.py
"""

import numpy as np
from collections import Counter


def main():
    Y = np.load("data/embeddings/train_Y.npy")  # (44906, 80)
    print(f"Training labels shape: {Y.shape}")

    # Treat each row as a tuple to count exact-match groups
    label_vectors = [tuple(row) for row in Y]
    counts = Counter(label_vectors)

    n_distinct = len(counts)
    n_total = len(label_vectors)
    group_sizes = np.array(list(counts.values()))

    print(f"\nTotal instances: {n_total}")
    print(f"Distinct label vectors: {n_distinct}")
    print(f"Mean instances per distinct vector: {group_sizes.mean():.2f}")
    print(f"Median instances per distinct vector: {np.median(group_sizes):.0f}")
    print(f"Vectors supported by only 1 instance: "
          f"{(group_sizes == 1).sum()} ({100 * (group_sizes == 1).sum() / n_distinct:.1f}% of distinct vectors)")
    print(f"Vectors supported by >=10 instances: "
          f"{(group_sizes >= 10).sum()} ({100 * (group_sizes >= 10).sum() / n_distinct:.1f}% of distinct vectors)")
    print(f"Largest group size: {group_sizes.max()}")

    # What fraction of ALL instances fall into "reliable" groups (>=10 support)?
    n_in_reliable = sum(c for c in counts.values() if c >= 10)
    print(f"\nInstances in reliably-supported groups (>=10): "
          f"{n_in_reliable} ({100 * n_in_reliable / n_total:.1f}% of all training instances)")


if __name__ == "__main__":
    main()