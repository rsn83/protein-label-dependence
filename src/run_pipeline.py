"""
Master pipeline: runs the full factorial —
  datasets (humloc, pcg, eukaryotego) x protocols (transductive, inductive)
  x encoders (gcn, sage) x heads (independent, dependency, gmnn, lamp, corgcn)

Every run uses compute_full_metrics — the ONE canonical metric set
(KDD 2025's 7 metrics + TMLR 2023's label homophily) — so every number in
every cell of this grid is directly comparable.

Usage:
    python src/run_pipeline.py
"""

import torch
import torch.nn.functional as F
import json
import os
import sys

sys.path.append(os.path.dirname(__file__))
from data_loaders import cache_all_datasets, load_cached_dataset, ALL_DATASET_NAMES, load_selected_labels
from model_registry import get_encoder, get_head, build_generic_dependency_edges, HEAD_TYPES, ENCODER_TYPES
from train_corgcn_precise import true_to_macro_targets, MultiLabelFocalLoss
from metrics import compute_full_metrics, format_metrics


def extract_induced_subgraph_edges(edge_index, mask):
    """Returns edge_index remapped to LOCAL indices, restricted to edges
    where both endpoints are in mask — needed so label_homophily's node
    indices align with a y array that's already been filtered by mask."""
    mask = mask.to(edge_index.device)  # fix: mask may be on CPU while edge_index is on GPU
    n = mask.shape[0]
    node_ids = mask.nonzero(as_tuple=True)[0]
    remap = torch.full((n,), -1, dtype=torch.long, device=mask.device)
    remap[node_ids] = torch.arange(len(node_ids), device=mask.device)

    src, dst = edge_index
    edge_in_mask = mask[src] & mask[dst]
    local_src = remap[src[edge_in_mask]]
    local_dst = remap[dst[edge_in_mask]]
    return torch.stack([local_src, local_dst])


def forward_pass(encoder, head, head_type, x, edge_index):
    z = encoder(x, edge_index)
    if head_type == "corgcn":
        logits, _ = head(z, edge_index)
    elif head_type == "gmnn":
        logits = head(z, edge_index)
    else:
        logits = head(z, edge_index) if getattr(head, "needs_edge_index", False) else head(z)
    return logits


def run_transductive(encoder_type, head_type, data, train_mask, val_mask, test_mask,
                      device, epochs=150, hidden_dim=64):
    in_dim = data.x.shape[1]
    num_labels = data.y.shape[1]
    x, edge_index, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)

    encoder = get_encoder(encoder_type, in_dim, hidden_dim, device)

    dep_edges = None
    if head_type == "dependency":
        dep_edges = build_generic_dependency_edges(y[train_mask].cpu().numpy())

    head, extra = get_head(head_type, hidden_dim, num_labels, device, dependency_edges=dep_edges)

    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=0.005 if encoder_type == "sage" else 0.001)

    focal_loss_fn = None
    if head_type == "corgcn":
        group_map = extra["group_map"]
        macro_y_train = true_to_macro_targets(y[train_mask], group_map, extra["k_prime"])
        class_weights = 1.0 / torch.sqrt(macro_y_train.sum(dim=0) + 1e-9)
        focal_loss_fn = MultiLabelFocalLoss(alpha=class_weights / class_weights.sum())

    best_val, best_test = -1.0, None
    for epoch in range(1, epochs + 1):
        encoder.train(); head.train()
        optimizer.zero_grad()
        z = encoder(x, edge_index)

        if head_type == "corgcn":
            logits, E_x = head(z, edge_index)
            macro_y = true_to_macro_targets(y, extra["group_map"], extra["k_prime"])
            L_cls = F.binary_cross_entropy_with_logits(logits[train_mask], y[train_mask])
            L_cmi = head.cmi_loss(E_x[train_mask], macro_y[train_mask])
            L_lm = head.lm_loss(E_x[train_mask], macro_y[train_mask], focal_loss_fn)
            alpha = (L_cls.detach() / (3 * L_cmi.detach() + 1e-8)).abs()
            beta = (L_cls.detach() / (3 * L_lm.detach() + 1e-8)).abs()
            loss = L_cls + alpha * L_cmi + beta * L_lm
        else:
            logits = forward_pass(encoder, head, head_type, x, edge_index)
            loss = F.binary_cross_entropy_with_logits(logits[train_mask], y[train_mask])

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            encoder.eval(); head.eval()
            z_eval = encoder(x, edge_index)
            if head_type == "corgcn":
                logits_eval, _ = head(z_eval, edge_index)
            else:
                logits_eval = forward_pass(encoder, head, head_type, x, edge_index)
            val_probs = torch.sigmoid(logits_eval[val_mask]).cpu().numpy()
            val_metrics = compute_full_metrics(y[val_mask].cpu().numpy(), val_probs)
            if val_metrics["micro_auc"] > best_val:
                best_val = val_metrics["micro_auc"]
                test_probs = torch.sigmoid(logits_eval[test_mask]).cpu().numpy()
                test_edge_index = extract_induced_subgraph_edges(edge_index, test_mask)
                best_test = compute_full_metrics(
                    y[test_mask].cpu().numpy(), test_probs,
                    edge_index=test_edge_index.cpu().numpy()
                )
    return best_test


def run_inductive(encoder_type, head_type, train_data, val_data, test_data,
                   device, epochs=150, hidden_dim=64):
    in_dim = train_data.x.shape[1]
    num_labels = train_data.y.shape[1]
    tx, tei, ty = train_data.x.to(device), train_data.edge_index.to(device), train_data.y.to(device)
    vx, vei, vy = val_data.x.to(device), val_data.edge_index.to(device), val_data.y.to(device)
    ttx, ttei, tty = test_data.x.to(device), test_data.edge_index.to(device), test_data.y.to(device)

    encoder = get_encoder(encoder_type, in_dim, hidden_dim, device)

    dep_edges = None
    if head_type == "dependency":
        dep_edges = build_generic_dependency_edges(ty.cpu().numpy())

    head, extra = get_head(head_type, hidden_dim, num_labels, device, dependency_edges=dep_edges)

    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=0.005 if encoder_type == "sage" else 0.001)

    focal_loss_fn = None
    if head_type == "corgcn":
        group_map = extra["group_map"]
        macro_y_train = true_to_macro_targets(ty, group_map, extra["k_prime"])
        class_weights = 1.0 / torch.sqrt(macro_y_train.sum(dim=0) + 1e-9)
        focal_loss_fn = MultiLabelFocalLoss(alpha=class_weights / class_weights.sum())

    best_val, best_test = -1.0, None
    for epoch in range(1, epochs + 1):
        encoder.train(); head.train()
        optimizer.zero_grad()

        if head_type == "corgcn":
            z = encoder(tx, tei)
            logits, E_x = head(z, tei)
            macro_y_train = true_to_macro_targets(ty, extra["group_map"], extra["k_prime"])
            L_cls = F.binary_cross_entropy_with_logits(logits, ty)
            L_cmi = head.cmi_loss(E_x, macro_y_train)
            L_lm = head.lm_loss(E_x, macro_y_train, focal_loss_fn)
            alpha = (L_cls.detach() / (3 * L_cmi.detach() + 1e-8)).abs()
            beta = (L_cls.detach() / (3 * L_lm.detach() + 1e-8)).abs()
            loss = L_cls + alpha * L_cmi + beta * L_lm
        else:
            logits = forward_pass(encoder, head, head_type, tx, tei)
            loss = F.binary_cross_entropy_with_logits(logits, ty)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            encoder.eval(); head.eval()
            if head_type == "corgcn":
                val_logits, _ = head(encoder(vx, vei), vei)
            else:
                val_logits = forward_pass(encoder, head, head_type, vx, vei)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_metrics = compute_full_metrics(vy.cpu().numpy(), val_probs)
            if val_metrics["micro_auc"] > best_val:
                best_val = val_metrics["micro_auc"]
                if head_type == "corgcn":
                    test_logits, _ = head(encoder(ttx, ttei), ttei)
                else:
                    test_logits = forward_pass(encoder, head, head_type, ttx, ttei)
                test_probs = torch.sigmoid(test_logits).cpu().numpy()
                best_test = compute_full_metrics(
                    tty.cpu().numpy(), test_probs, edge_index=ttei.cpu().numpy()
                )
    return best_test


import numpy as np


def aggregate_trials(trial_results):
    """Given a list of metric dicts (one per trial), returns mean+std per metric."""
    keys = trial_results[0].keys()
    agg = {}
    for k in keys:
        vals = [r[k] for r in trial_results if k in r]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    agg["n_trials"] = len(trial_results)
    return agg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nCaching datasets (skips already-cached ones)...")
    selected_indices = load_selected_labels()
    cache_all_datasets(selected_indices=selected_indices)

    datasets = ALL_DATASET_NAMES
    protocols = ["transductive", "inductive"]
    encoders = ENCODER_TYPES
    heads = HEAD_TYPES
    n_trials = 5  # matches Bei et al.'s own "five trial runs" protocol

    os.makedirs("results", exist_ok=True)
    results_path = "results/full_pipeline_results_5trial.json"

    all_results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Resuming — {len(all_results)} configs already completed, will skip those.")

    total_configs = len(datasets) * len(protocols) * len(encoders) * len(heads)
    config_num = 0

    for dataset_name in datasets:
        for protocol in protocols:
            for encoder_type in encoders:
                for head_type in heads:
                    config_num += 1
                    key = f"{dataset_name}__{protocol}__{encoder_type}__{head_type}"

                    if key in all_results and "error" not in all_results[key]:
                        print(f"\n[{config_num}/{total_configs}] {key} — already done, skipping")
                        continue

                    print(f"\n[{config_num}/{total_configs}] {key} — running {n_trials} trials")
                    trial_results = []
                    try:
                        for trial in range(n_trials):
                            if protocol == "transductive":
                                # Note: for datasets with a fixed authors' split.pt (Humloc,
                                # EukaryoteGo, Blogcatalog), all trials currently reuse that
                                # SAME split — only model initialization/training randomness
                                # varies across trials. For pcg (no split.pt) and the
                                # inductive-ized partitions, each trial gets an independent
                                # random split (seed=trial), closer to the paper's true
                                # "5 independent splits" protocol.
                                data, train_mask, val_mask, test_mask = load_cached_dataset(dataset_name, "transductive")
                                torch.manual_seed(trial)
                                result = run_transductive(encoder_type, head_type, data,
                                                           train_mask, val_mask, test_mask, device)
                            else:
                                train_data, val_data, test_data = load_cached_dataset(dataset_name, "inductive")
                                torch.manual_seed(trial)
                                result = run_inductive(encoder_type, head_type,
                                                        train_data, val_data, test_data, device)
                            trial_results.append(result)
                            print(f"  trial {trial}: micro_auc={result['micro_auc']:.4f}")

                        aggregated = aggregate_trials(trial_results)
                        all_results[key] = aggregated
                        print(f"  MEAN micro_auc: {aggregated['micro_auc_mean']:.4f} "
                              f"(+/- {aggregated['micro_auc_std']:.4f})")
                    except Exception as e:
                        print(f"  FAILED: {e}")
                        all_results[key] = {"error": str(e)}

                    with open(results_path, "w") as f:
                        json.dump(all_results, f, indent=2)

    print(f"\nDone. Saved all {len(all_results)} configs to {results_path}")


if __name__ == "__main__":
    main()
