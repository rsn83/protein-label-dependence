"""
Select the top-N most frequent labels from the 121-label GO-term set.

Structure learning over all 121 labels was empirically confirmed intractable
(test_structure_learning_scale.py: 80 labels = 53 sec, 121 labels = 87+ hour
projection). This selects a frequency-ranked subset, saved once so every
method in the comparison (baseline, mixture model) trains/evaluates on the
exact same label subset — required for a fair comparison.

Usage:
    python src/label_selection.py --top_n 80
"""

import argparse
import json
import numpy as np
from torch_geometric.datasets import PPI


def select_top_n_labels(top_n: int = 80, root: str = "./data/raw"):
    train_dataset = PPI(root=root, split="train")
    all_labels = np.concatenate(
        [g.y.numpy() for g in train_dataset], axis=0
    )  # (num_instances, 121)

    label_frequency = all_labels.sum(axis=0)  # count of positive instances per label
    ranked_indices = np.argsort(-label_frequency)  # descending frequency
    selected = sorted(ranked_indices[:top_n].tolist())  # keep sorted for consistent ordering

    print(f"Selected top {top_n} of {all_labels.shape[1]} labels by frequency.")
    print(f"Frequency range in selection: "
          f"{label_frequency[selected].min():.0f} to {label_frequency[selected].max():.0f} "
          f"positive instances (out of {all_labels.shape[0]} total).")
    print(f"Excluded {all_labels.shape[1] - top_n} lowest-frequency labels, "
          f"min excluded frequency: {label_frequency[ranked_indices[top_n:]].max():.0f}.")

    with open("data/selected_labels.json", "w") as f:
        json.dump(
            {
                "top_n": top_n,
                "total_labels": int(all_labels.shape[1]),
                "selected_indices": selected,
                "note": "Structure learning over all 121 labels was empirically "
                        "intractable (see results/scale_test output). This subset "
                        "of the top-N most frequent labels is used for all methods "
                        "in the comparison.",
            },
            f,
            indent=2,
        )
    print("Saved selection to data/selected_labels.json")
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_n", type=int, default=80)
    args = parser.parse_args()
    select_top_n_labels(top_n=args.top_n)