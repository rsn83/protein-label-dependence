"""
Smoke test: runs every head type x both encoders x both protocols, on
Humloc only (smallest dataset), with few epochs — NOT for real results,
just to confirm the full pipeline machinery works end-to-end before
committing to the full 5-dataset x 2-protocol x 2-encoder x 5-head sweep.

Usage:
    python src/smoke_test.py
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))
from run_pipeline import run_transductive, run_inductive
from data_loaders import cache_all_datasets, load_cached_dataset, load_selected_labels
from model_registry import HEAD_TYPES, ENCODER_TYPES
from metrics import format_metrics
import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    selected_indices = load_selected_labels()
    cache_all_datasets(selected_indices=selected_indices)

    dataset_name = "humloc"
    protocols = ["transductive", "inductive"]
    encoders = ENCODER_TYPES
    heads = HEAD_TYPES
    smoke_epochs = 20

    failures = []
    successes = []
    total = len(protocols) * len(encoders) * len(heads)
    run_num = 0

    for protocol in protocols:
        for encoder_type in encoders:
            for head_type in heads:
                run_num += 1
                key = f"{dataset_name}__{protocol}__{encoder_type}__{head_type}"
                print(f"\n[{run_num}/{total}] {key}")
                try:
                    if protocol == "transductive":
                        data, train_mask, val_mask, test_mask = load_cached_dataset(dataset_name, "transductive")
                        result = run_transductive(encoder_type, head_type, data,
                                                   train_mask, val_mask, test_mask, device,
                                                   epochs=smoke_epochs)
                    else:
                        train_data, val_data, test_data = load_cached_dataset(dataset_name, "inductive")
                        result = run_inductive(encoder_type, head_type,
                                                train_data, val_data, test_data, device,
                                                epochs=smoke_epochs)
                    print(f"  OK")
                    print(format_metrics(result))
                    successes.append(key)
                except Exception as e:
                    print(f"  FAILED: {type(e).__name__}: {e}")
                    failures.append((key, str(e)))

    print(f"\n\n=== SMOKE TEST SUMMARY ===")
    print(f"Passed: {len(successes)}/{total}")
    print(f"Failed: {len(failures)}/{total}")
    if failures:
        print("\nFailures:")
        for key, err in failures:
            print(f"  {key}: {err}")
    else:
        print("\nAll combinations ran cleanly. Safe to proceed to the full sweep.")


if __name__ == "__main__":
    main()
