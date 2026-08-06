"""
Shared multi-label evaluation metrics.

Used identically by every method in the comparison (baseline, your
mixture model, and any later additions) so results are directly
comparable — same metric code, same thresholding, no drift between runs.
"""

import numpy as np
from sklearn.metrics import f1_score, jaccard_score, roc_auc_score, accuracy_score


def compute_all_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    y_true: (N, num_labels) binary ground truth
    y_prob: (N, num_labels) predicted probabilities (post-sigmoid, pre-threshold)
    """
    y_pred = (y_prob > threshold).astype(int)

    metrics = {}

    metrics["micro_f1"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    metrics["jaccard_micro"] = jaccard_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["jaccard_macro"] = jaccard_score(y_true, y_pred, average="macro", zero_division=0)

    # Subset accuracy: fraction of instances with an EXACT full label-set match.
    # Typically low for 121 labels — reported for completeness, not as the headline number.
    metrics["subset_accuracy"] = accuracy_score(y_true, y_pred)

    # ROC-AUC needs raw probabilities, not thresholded predictions.
    # Computed per-label then averaged; skips labels with only one class present
    # in y_true (undefined AUC otherwise), and reports how many were skipped.
    valid_labels = [
        k for k in range(y_true.shape[1])
        if len(np.unique(y_true[:, k])) == 2
    ]
    if valid_labels:
        metrics["roc_auc_macro"] = roc_auc_score(
            y_true[:, valid_labels], y_prob[:, valid_labels], average="macro"
        )
    else:
        metrics["roc_auc_macro"] = float("nan")
    metrics["roc_auc_labels_skipped"] = y_true.shape[1] - len(valid_labels)

    return metrics


def format_metrics(metrics: dict) -> str:
    lines = []
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.4f}")
        else:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)