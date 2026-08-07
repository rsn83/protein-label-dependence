"""
Timing check: run EM on a small subset (few label vectors, few features)
to estimate real full-scale runtime before committing to it.

Usage:
    python src/time_em_subset.py
"""

import numpy as np
import time
import sys
import os

sys.path.append(os.path.dirname(__file__))
from fit_mixture_model import load_data, estimate_alpha, estimate_phi, em_estimate_theta


def main():
    Z, Y, lds, parents = load_data()

    # Small subset: first 2000 instances, first 20 features
    n_sub = 2000
    n_feat_sub = 20
    Z_sub = Z[:n_sub, :n_feat_sub]
    Y_sub = Y[:n_sub]

    print(f"Timing subset: {n_sub} instances, {n_feat_sub} features (full: "
          f"{Z.shape[0]} instances, {Z.shape[1]} features)")

    start = time.time()
    phi_sub = estimate_phi(Z_sub, Y_sub, lds, n_bins=4)
    phi_time = time.time() - start
    print(f"phi estimation (subset): {phi_time:.1f} sec")

    start = time.time()
    theta_sub = em_estimate_theta(Z_sub, Y_sub, lds, phi_sub, n_iters=2, tol=0.05)
    em_time = time.time() - start
    print(f"EM, 2 iterations (subset): {em_time:.1f} sec")

    # Extrapolate to full scale
    scale_factor_instances = Z.shape[0] / n_sub
    scale_factor_features = Z.shape[1] / n_feat_sub
    full_phi_estimate = phi_time * scale_factor_features * scale_factor_instances
    full_em_per_iter_estimate = (em_time / 2) * scale_factor_features * scale_factor_instances

    print(f"\nExtrapolated full-scale estimate:")
    print(f"  phi estimation: ~{full_phi_estimate / 60:.1f} minutes")
    print(f"  EM per iteration: ~{full_em_per_iter_estimate / 60:.1f} minutes "
          f"(x up to 20 iterations = ~{full_em_per_iter_estimate * 20 / 3600:.1f} hours)")


if __name__ == "__main__":
    main()