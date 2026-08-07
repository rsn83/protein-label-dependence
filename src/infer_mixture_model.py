"""
Step 3 of the ECAI mixture model: inference on held-out data.

Implements Figure 5 from the paper: for each label dependency set LS_i,
find the value assignment to L_i and its parents that maximizes
Pr(L_i, Pa(L_i) | features), iterating until no further improvement.

Usage:
    python src/infer_mixture_model.py --split test
"""

import numpy as np
import pickle
import argparse
import sys
import os

sys.path.append(os.path.dirname(__file__))
from fit_mixture_model import get_feature_given_lds_prob
from metrics import compute_all_metrics, format_metrics


def load_model():
    with open("checkpoints/mixture_model_params_gaussian.pkl", "rb") as f:
        return pickle.load(f)


FALLBACK_COUNTS = {"phi_fallback": 0, "phi_total": 0,
                   "alpha_fallback": 0, "alpha_total": 0}


def get_alpha_prob(label_idx, parent_values, alpha_table):
    """Pr(L_i = 1 | Pa(L_i) = parent_values), with fallback for unseen combos."""
    table = alpha_table[label_idx]
    FALLBACK_COUNTS["alpha_total"] += 1
    if parent_values in table:
        return table[parent_values]
    FALLBACK_COUNTS["alpha_fallback"] += 1
    return 0.5  # unseen parent combination: uninformative fallback


def get_normalized_theta_vector(j, candidate_full, lds, theta_raw, n_labels):
    """
    Gathers raw theta weights for feature j across all k, using each k's own
    (small) LDS-projection of the candidate full vector, then normalizes to
    sum to 1 — reconstructing the mixture-weight vector at use time from the
    small, well-supported per-(j,k) tables (see reworked em_estimate_theta).
    """
    raw = np.zeros(n_labels)
    for k in range(n_labels):
        lsk_indices = sorted(lds[k])
        combo = tuple(candidate_full[idx] for idx in lsk_indices)
        raw[k] = theta_raw[j][k].get(combo, 1.0 / n_labels)
    total = raw.sum()
    return raw / total if total > 0 else np.full(n_labels, 1.0 / n_labels)


def score_candidate_full_vector(instance_features, candidate_full, model, feature_dim, n_labels):
    """
    Computes the FULL log-joint log Pr(l, f) per Equation 3, for one complete
    candidate label vector. This is the correct basis for Figure 5's conditional
    probability: since other labels and features are held fixed while comparing
    candidates for one LDS, Pr(LDS candidate | f, other labels) is proportional
    to this full joint (normalizing constant is identical across candidates).

    theta is now reconstructed per feature via the small per-(j,k) LDS-projected
    tables (reworked EM), not a full-vector lookup — see get_normalized_theta_vector.
    """
    alpha = model["alpha"]
    phi = model["phi"]
    parents = model["parents"]
    theta_raw = model["theta"]
    lds = model["lds"]

    log_score = 0.0
    for label_idx in range(n_labels):
        pa_indices = parents[label_idx]
        parent_vals = tuple(candidate_full[p] for p in pa_indices)
        p = get_alpha_prob(label_idx, parent_vals, alpha)
        p = p if candidate_full[label_idx] == 1 else (1 - p)
        log_score += np.log(max(p, 1e-6))

    global_stats = model["global_stats"]
    for j in range(feature_dim):
        theta_vec = get_normalized_theta_vector(j, candidate_full, lds, theta_raw, n_labels)
        mixture_sum = 0.0
        for k in range(n_labels):
            lsk_indices = sorted(lds[k])
            lsk_combo = tuple(candidate_full[idx] for idx in lsk_indices)
            p_feat_given_k = get_feature_given_lds_prob(
                instance_features[j], lsk_combo, phi[(j, k)], FALLBACK_COUNTS,
                global_stats=global_stats, feature_idx=j
            )
            mixture_sum += theta_vec[k] * p_feat_given_k
        log_score += np.log(max(mixture_sum, 1e-10))

    return log_score


def infer_labels(Z: np.ndarray, model: dict, n_labels: int, max_outer_iters: int = 3):
    """
    Figure 5 procedure: for each instance, iterate over label dependency sets,
    assign the value combination maximizing the joint score (now properly
    using theta), repeat until stable.
    """
    lds = model["lds"]
    n_instances, feature_dim = Z.shape
    predictions = np.zeros((n_instances, n_labels), dtype=int)

    for outer_iter in range(max_outer_iters):
        changed = 0
        for i in range(n_instances):
            if i % 1000 == 0:
                print(f"  outer iter {outer_iter + 1}, instance {i}/{n_instances}")
            for label_idx in range(n_labels):
                lds_indices = sorted(lds[label_idx])
                m = len(lds_indices)
                best_score = -np.inf
                best_assignment = None
                for combo_bits in range(2 ** m):
                    assignment = tuple((combo_bits >> b) & 1 for b in range(m))
                    candidate_full = list(predictions[i])
                    for pos, idx in enumerate(lds_indices):
                        candidate_full[idx] = assignment[pos]
                    score = score_candidate_full_vector(
                        Z[i], candidate_full, model, feature_dim, n_labels
                    )
                    if score > best_score:
                        best_score = score
                        best_assignment = assignment
                for pos, label_pos in enumerate(lds_indices):
                    if predictions[i, label_pos] != best_assignment[pos]:
                        changed += 1
                    predictions[i, label_pos] = best_assignment[pos]
        print(f"Outer iteration {outer_iter + 1}: {changed} label values changed.")
        if changed == 0:
            print("Converged.")
            break

    return predictions


def main(split: str, n_subset: int = None):
    model = load_model()
    Z = np.load(f"data/embeddings/{split}_Z.npy")  # CONTINUOUS — Gaussian variant
    Y = np.load(f"data/embeddings/{split}_Y.npy")
    if n_subset:
        Z = Z[:n_subset]
        Y = Y[:n_subset]
    n_labels = Y.shape[1]

    print(f"Running inference (GAUSSIAN variant) on {split} split: {Z.shape[0]} instances, {n_labels} labels")
    predictions = infer_labels(Z, model, n_labels)

    # Convert hard predictions to "probabilities" of 0/1 for the shared metrics function
    probs = predictions.astype(float)
    metrics = compute_all_metrics(Y, probs, threshold=0.5)

    print(f"\n{split} metrics:")
    print(format_metrics(metrics))

    print(f"\nFallback diagnostics:")
    for name in ["phi", "alpha"]:
        total = FALLBACK_COUNTS[f"{name}_total"]
        fb = FALLBACK_COUNTS[f"{name}_fallback"]
        rate = 100 * fb / total if total > 0 else 0
        print(f"  {name}: {fb}/{total} fallback ({rate:.1f}%)")

    pred_density = predictions.mean()
    true_density = Y.mean()
    print(f"\nLabel density — predicted: {pred_density:.4f}, actual: {true_density:.4f}")

    import json
    os.makedirs("results", exist_ok=True)
    with open(f"results/mixture_model_gaussian_{split}_results.json", "w") as f:
        json.dump({"method": "GraphSAGE + ECAI 2016 mixture model", "metrics": metrics}, f, indent=2)
    print(f"Saved to results/mixture_model_{split}_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--n_subset", type=int, default=None, help="Run on only N instances, for timing")
    args = parser.parse_args()
    main(args.split, args.n_subset)