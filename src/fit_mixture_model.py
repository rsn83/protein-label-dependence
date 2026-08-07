"""
Step 2 of the ECAI mixture model: EM parameter estimation.

Implements Equation 3 and the E-step/M-step from Section 3.1 of the paper:
  - alpha_i   = Pr(L_i | Pa(L_i))                       — label conditional probs
  - phi_{j,k} = Pr(F_j | LDS_k)                          — feature-given-LDS probs
  - theta^l_{j,k} = Pr(lambda_Fj = k | l)                — mixture weights (EM-estimated)

All with Laplace smoothing throughout, as specified in the paper.

WARNING: the EM step as literally specified in the paper (looping per distinct
label vector x per feature x per k) is computationally heavy at this scale
(3577 distinct label vectors x 256 features x 80 k-values). This script runs
it but may take a long time — run the small-scale timing check first
(see fit_mixture_model_timing_check.py) before running this on full data.

Usage:
    python src/fit_mixture_model.py
"""

import numpy as np
import json
import pickle
from collections import defaultdict


LAPLACE_ALPHA = 1.0  # Laplace smoothing pseudocount


def load_data():
    Z = np.load("data/embeddings/train_Z.npy")  # (n, 256), CONTINUOUS — Gaussian variant
    Y = np.load("data/embeddings/train_Y.npy").astype(int)  # (n, 80)
    with open("data/label_structure.json") as f:
        structure = json.load(f)
    lds = {int(k): v for k, v in structure["label_dependency_sets"].items()}
    parents = {int(k): v for k, v in structure["parents"].items()}
    return Z, Y, lds, parents


def estimate_alpha(Y: np.ndarray, parents: dict):
    """
    alpha_i = Pr(L_i = 1 | Pa(L_i)) for every combination of parent values.
    Returns: dict label_i -> dict{parent_value_tuple: prob_of_1}
    """
    n_labels = Y.shape[1]
    alpha = {}
    for i in range(n_labels):
        pa = parents[i]
        if len(pa) == 0:
            p1 = (Y[:, i].sum() + LAPLACE_ALPHA) / (Y.shape[0] + 2 * LAPLACE_ALPHA)
            alpha[i] = {(): p1}
            continue
        pa_values = Y[:, pa]
        table = {}
        unique_combos = set(map(tuple, pa_values))
        for combo in unique_combos:
            mask = np.all(pa_values == np.array(combo), axis=1)
            n_match = mask.sum()
            n_pos = Y[mask, i].sum()
            table[combo] = (n_pos + LAPLACE_ALPHA) / (n_match + 2 * LAPLACE_ALPHA)
        alpha[i] = table
    return alpha


MIN_VARIANCE = 1e-2  # variance floor, prevents degenerate near-zero variance for small groups


def estimate_phi(Z: np.ndarray, Y: np.ndarray, lds: dict):
    """
    GAUSSIAN VARIANT: phi_{j,k}(v) = N(v; mu_{j,k,combo}, sigma^2_{j,k,combo})
    Operates on continuous Z directly (no discretization). Returns:
    dict (j, k) -> dict{ldsk_value_tuple: (mean, variance)}
    """
    n_features = Z.shape[1]
    n_labels = Y.shape[1]
    phi = {}
    for k in range(n_labels):
        lsk = lds[k]
        lsk_values = Y[:, lsk]
        unique_combos = set(map(tuple, lsk_values))
        for j in range(n_features):
            table = {}
            for combo in unique_combos:
                mask = np.all(lsk_values == np.array(combo), axis=1)
                vals = Z[mask, j]
                n = len(vals)
                mu = vals.mean() if n > 0 else 0.0
                var = vals.var() if n > 1 else MIN_VARIANCE
                var = max(var, MIN_VARIANCE)  # floor, avoids degenerate near-zero variance
                table[combo] = (mu, var)
            phi[(j, k)] = table
    return phi


def compute_global_feature_stats(Z: np.ndarray):
    """Global per-feature mean/variance, used as the fallback for unseen LDS combos."""
    return {j: (Z[:, j].mean(), max(Z[:, j].var(), MIN_VARIANCE)) for j in range(Z.shape[1])}


def get_feature_given_lds_prob(f_val, lds_combo, phi_table, counter=None, global_stats=None, feature_idx=None):
    """GAUSSIAN VARIANT: evaluates the Gaussian PDF, not a discrete bin lookup."""
    if lds_combo in phi_table:
        if counter is not None:
            counter["phi_total"] += 1
        mu, var = phi_table[lds_combo]
    else:
        if counter is not None:
            counter["phi_total"] += 1
            counter["phi_fallback"] += 1
        if global_stats is not None and feature_idx is not None:
            mu, var = global_stats[feature_idx]
        else:
            mu, var = 0.0, 1.0  # last-resort fallback if no global stats provided
    density = (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-((f_val - mu) ** 2) / (2 * var))
    return density


def em_estimate_theta(Z: np.ndarray, Y: np.ndarray, lds: dict, phi: dict, global_stats: dict,
                       n_iters: int = 5, tol: float = 0.05):
    """
    EM for theta, REWORKED: theta_{j,k} is now conditioned on the values of
    labels WITHIN LDS_k (<=3 labels, <=8 combos) instead of the full 80-label
    vector. This resolves the 94% fallback rate observed with full-vector
    conditioning (see diagnostic run) and is more consistent with the model's
    own conditional-independence assumption (Eq. 2): a feature should be
    explainable by its LDS's values alone.

    Structure: theta_raw[j][k] = dict{lds_k_combo: raw_weight}. At use time
    (both here and at inference), the raw weights for a specific full label
    vector l are gathered as theta_raw[j][k][combo_k(l)] for each k, then
    normalized to sum to 1 across k — same mixture semantics as the paper,
    smaller and well-supported conditioning variable.
    """
    n_instances, n_features = Z.shape
    n_labels = Y.shape[1]
    lds_indices_list = [sorted(lds[k]) for k in range(n_labels)]

    # Precompute each instance's LDS-projection combo for every k, once.
    print("Precomputing per-instance LDS projections...")
    all_combos = np.zeros((n_instances, n_labels), dtype=object)
    for idx in range(n_instances):
        l = Y[idx]
        for k in range(n_labels):
            all_combos[idx, k] = tuple(l[lds_indices_list[k]])

    # Initialize theta_raw uniformly (every combo starts at 1/n_labels, matching
    # the original paper's uniform EM initialization).
    theta_raw = {j: {k: {} for k in range(n_labels)} for j in range(n_features)}

    for iteration in range(n_iters):
        resp_sum = {j: {k: defaultdict(float) for k in range(n_labels)} for j in range(n_features)}
        resp_count = {j: {k: defaultdict(int) for k in range(n_labels)} for j in range(n_features)}
        max_change = 0.0

        for idx in range(n_instances):
            if idx % 10000 == 0:
                print(f"  iter {iteration + 1}, instance {idx}/{n_instances}")
            combos = all_combos[idx]
            for j in range(n_features):
                raw = np.array([theta_raw[j][k].get(combos[k], 1.0 / n_labels) for k in range(n_labels)])
                raw_sum = raw.sum()
                prior = raw / raw_sum if raw_sum > 0 else np.full(n_labels, 1.0 / n_labels)

                likelihood = np.array([
                    get_feature_given_lds_prob(Z[idx, j], combos[k], phi[(j, k)],
                                                global_stats=global_stats, feature_idx=j)
                    for k in range(n_labels)
                ])
                resp = prior * likelihood
                resp_total = resp.sum()
                if resp_total > 0:
                    resp = resp / resp_total

                for k in range(n_labels):
                    resp_sum[j][k][combos[k]] += resp[k]
                    resp_count[j][k][combos[k]] += 1

        for j in range(n_features):
            for k in range(n_labels):
                for combo, total_resp in resp_sum[j][k].items():
                    count = resp_count[j][k][combo]
                    new_val = (total_resp + LAPLACE_ALPHA / n_labels) / (count + LAPLACE_ALPHA)
                    old_val = theta_raw[j][k].get(combo, 1.0 / n_labels)
                    max_change = max(max_change, abs(new_val - old_val))
                    theta_raw[j][k][combo] = new_val

        print(f"EM iteration {iteration + 1}: max parameter change = {max_change:.4f}")
        if max_change < tol:
            print("Converged.")
            break

    return theta_raw


def main():
    Z, Y, lds, parents = load_data()
    print(f"Loaded: Z {Z.shape}, Y {Y.shape}, {len(lds)} label dependency sets")
    print("GAUSSIAN VARIANT: using continuous embeddings directly, no discretization.")

    print("\nEstimating alpha (label | parents)...")
    alpha = estimate_alpha(Y, parents)

    print("Estimating phi (feature | LDS) as Gaussian mean/variance...")
    phi = estimate_phi(Z, Y, lds)

    global_stats = compute_global_feature_stats(Z)

    print("\nRunning EM for theta (mixture weights)...")
    print("(Bounded at 5 iterations, ~2 hr worst case. Subset test converged in 1 "
          "iteration, so this will likely finish much sooner via early stopping.)")
    theta = em_estimate_theta(Z, Y, lds, phi, global_stats, n_iters=5, tol=0.05)

    with open("checkpoints/mixture_model_params_gaussian.pkl", "wb") as f:
        pickle.dump(
            {"alpha": alpha, "phi": phi, "theta": theta, "lds": lds,
             "parents": parents, "global_stats": global_stats, "variant": "gaussian"},
            f,
        )
    print("\nSaved fitted parameters to checkpoints/mixture_model_params_gaussian.pkl")


if __name__ == "__main__":
    main()