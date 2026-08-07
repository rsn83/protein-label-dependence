"""
Shared multi-label evaluation metrics.

Used identically by every method in the comparison (baseline, your
mixture model, and any later additions) so results are directly
comparable — same metric code, same thresholding, no drift between runs.
"""

import numpy as np
from sklearn.metrics import (
    f1_score, jaccard_score, roc_auc_score, accuracy_score,
    average_precision_score, coverage_error, label_ranking_loss,
    label_ranking_average_precision_score, hamming_loss
)


def compute_corgcn_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    The 7 metrics used in Bei et al. (KDD 2025) Table 2, for direct comparison
    against their reported numbers. Uses raw probabilities (not thresholded)
    except Hamming Loss, which needs a 0.5 threshold per their convention.
    """
    y_pred = (y_prob > 0.5).astype(int)
    metrics = {}
    metrics["ranking_loss"] = label_ranking_loss(y_true, y_prob)
    metrics["hamming_loss"] = hamming_loss(y_true, y_pred)

    valid_labels = [k for k in range(y_true.shape[1]) if len(np.unique(y_true[:, k])) == 2]
    if valid_labels:
        metrics["macro_auc"] = roc_auc_score(y_true[:, valid_labels], y_prob[:, valid_labels], average="macro")
        metrics["micro_auc"] = roc_auc_score(y_true[:, valid_labels], y_prob[:, valid_labels], average="micro")
        metrics["macro_ap"] = average_precision_score(y_true[:, valid_labels], y_prob[:, valid_labels], average="macro")
        metrics["micro_ap"] = average_precision_score(y_true[:, valid_labels], y_prob[:, valid_labels], average="micro")
    else:
        metrics["macro_auc"] = metrics["micro_auc"] = metrics["macro_ap"] = metrics["micro_ap"] = float("nan")

    metrics["lrap"] = label_ranking_average_precision_score(y_true, y_prob)
    return metrics


def compute_full_metrics(y_true: np.ndarray, y_prob: np.ndarray, edge_index: np.ndarray = None) -> dict:
    """
    THE canonical evaluation function, combining:
      - Bei et al. (KDD 2025) Table 2's 7 metrics: ranking_loss, hamming_loss,
        macro/micro_auc, macro/micro_ap, lrap
      - Zhao et al. (TMLR 2023)'s label homophily metric (only computed if
        edge_index is provided — it's a property of the graph, not predictions)

    Use this for ALL evaluations from here forward, across every dataset and
    method, so every reported number is on a fixed, consistent, literature-
    grounded yardstick.
    """
    metrics = compute_corgcn_metrics(y_true, y_prob)
    if edge_index is not None:
        metrics["label_homophily"] = compute_label_homophily(edge_index, y_true)
    return metrics


def compute_label_homophily(edge_index: np.ndarray, y: np.ndarray) -> float:
    """
    Zhao et al. (2023, TMLR) label homophily metric: average Jaccard similarity
    of label sets between connected nodes.
    h = (1/|E|) * sum_{(i,j) in E} |Y_i ∩ Y_j| / |Y_i ∪ Y_j|
    edge_index: (2, E) array of node index pairs.
    """
    src, dst = edge_index[0], edge_index[1]
    y_src = y[src].astype(bool)
    y_dst = y[dst].astype(bool)
    intersection = (y_src & y_dst).sum(axis=1)
    union = (y_src | y_dst).sum(axis=1)
    valid = union > 0
    if valid.sum() == 0:
        return 0.0
    jaccard_per_edge = intersection[valid] / union[valid]
    return float(jaccard_per_edge.mean())


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
