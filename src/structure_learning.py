"""
Step 1 of the ECAI mixture model: learn Bayesian network structure over labels.

Uses the same HillClimbSearch approach already validated as tractable at
80 labels (~53 sec). Produces the label dependency sets LS_i = {L_i} U Pa(L_i)
that the rest of the model (alpha, phi, theta estimation, inference) depends on.

Usage:
    python src/structure_learning.py
"""

import numpy as np
import pandas as pd
import json
import time
from pgmpy.estimators import HillClimbSearch


def learn_structure(Y: np.ndarray, max_indegree: int = 2):
    """
    Y: (n_instances, n_labels) binary label matrix.
    Returns: dict mapping label index -> list of parent label indices.
    """
    n_labels = Y.shape[1]
    col_names = [f"L{i}" for i in range(n_labels)]
    df = pd.DataFrame(Y.astype(int), columns=col_names).astype("category")

    print(f"Learning BN structure over {n_labels} labels ({Y.shape[0]} instances)...")
    start = time.time()
    hc = HillClimbSearch(df)
    model = hc.estimate(scoring_method="bic-d", max_indegree=max_indegree, max_iter=150)
    elapsed = time.time() - start
    print(f"Structure learning completed in {elapsed:.1f} sec. "
          f"Learned {len(model.edges())} edges.")

    parents = {i: [] for i in range(n_labels)}
    for parent_name, child_name in model.edges():
        parent_idx = int(parent_name[1:])  # "L5" -> 5
        child_idx = int(child_name[1:])
        parents[child_idx].append(parent_idx)

    return parents


def build_label_dependency_sets(parents: dict):
    """LS_i = {L_i} U Pa(L_i), as defined in the ECAI paper Section 2.1."""
    lds = {}
    for i, pa in parents.items():
        lds[i] = sorted([i] + pa)  # label itself plus its parents
    return lds


def main():
    Y_train = np.load("data/embeddings/train_Y.npy")

    parents = learn_structure(Y_train, max_indegree=2)
    lds = build_label_dependency_sets(parents)

    n_with_parents = sum(1 for p in parents.values() if len(p) > 0)
    print(f"\n{n_with_parents} of {len(parents)} labels have at least one parent.")
    lds_sizes = [len(s) for s in lds.values()]
    print(f"Label dependency set sizes: min {min(lds_sizes)}, "
          f"max {max(lds_sizes)}, mean {np.mean(lds_sizes):.2f}")

    with open("data/label_structure.json", "w") as f:
        json.dump(
            {
                "parents": {str(k): v for k, v in parents.items()},
                "label_dependency_sets": {str(k): v for k, v in lds.items()},
                "max_indegree": 2,
                "max_iter": 150,
                "note": "Per-iteration cost grows with graph density during search "
                        "(observed degrading from ~1.6 sec/iter to ~34 sec/iter as edges "
                        "accumulated). Capped at 150 iterations (bounded, known runtime) "
                        "rather than run to full convergence — an approximate structure, "
                        "not a global optimum. Documented scoping decision.",
            },
            f,
            indent=2,
        )
    print("Saved structure to data/label_structure.json")


if __name__ == "__main__":
    main()