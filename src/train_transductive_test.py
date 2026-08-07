"""
Transductive test: single graph, node-level train/val/test split (60/20/20),
matching Zhao et al. (2023) / CorGCN's own protocol.

Tests whether today's findings (replace-vs-augment; label-graph-vs-instance-
graph) hold when the FULL graph structure (all nodes, all edges) is visible
throughout training — only train-node LABELS are hidden from the loss, not
the graph itself. This is the genuine transductive setting, as opposed to
SagePPI's native inductive (separate train/val/test graphs) setup.

Runs baseline, GMNN-inspired (best inductive augment method), and CorGCN
(best inductive replace method) on the SAME single graph for direct
comparison.

Usage:
    python src/train_transductive_test.py
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import PPI
import json
import os
import sys

sys.path.append(os.path.dirname(__file__))
from train_baseline import GraphSAGEEncoder, IndependentLabelHead, load_selected_labels
from train_gmnn_style_baseline import GMNNStyleHead
from train_corgcn_precise import (
    CorGCNHead, MultiLabelFocalLoss, build_macro_group_map,
    true_to_macro_targets, compute_focal_alpha
)
from metrics import compute_all_metrics, format_metrics


def get_single_graph(selected_indices, min_nodes=1500):
    train_dataset = PPI(root="./data/raw", split="train")
    for g in train_dataset:
        if g.num_nodes >= min_nodes:
            g = g.clone()
            idx = torch.tensor(selected_indices, dtype=torch.long)
            g.y = g.y[:, idx]
            return g
    raise ValueError("No graph found meeting min_nodes requirement")


def make_transductive_masks(n, seed=0, train_frac=0.6, val_frac=0.2):
    torch.manual_seed(seed)
    perm = torch.randperm(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True
    return train_mask, val_mask, test_mask


@torch.no_grad()
def evaluate_masked(encoder, head, data, mask, device, needs_edge_index=False):
    encoder.eval()
    head.eval()
    x, edge_index, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)
    z = encoder(x, edge_index)
    if needs_edge_index:
        out = head(z, edge_index)
        logits = out[0] if isinstance(out, tuple) else out
    else:
        logits = head(z)
    probs = torch.sigmoid(logits[mask]).cpu().numpy()
    ys = y[mask].cpu().numpy()
    return compute_all_metrics(ys, probs)


def run_independent(data, train_mask, val_mask, test_mask, device, epochs=100):
    in_dim = data.x.shape[1]
    num_labels = data.y.shape[1]
    encoder = GraphSAGEEncoder(in_dim, 256, 256).to(device)
    head = IndependentLabelHead(256, num_labels).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=0.005)

    x, edge_index, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)
    best_val, best_test = -1.0, None
    for epoch in range(1, epochs + 1):
        encoder.train(); head.train()
        optimizer.zero_grad()
        z = encoder(x, edge_index)
        logits = head(z)
        loss = F.binary_cross_entropy_with_logits(logits[train_mask], y[train_mask])
        loss.backward(); optimizer.step()

        val_metrics = evaluate_masked(encoder, head, data, val_mask, device)
        if val_metrics["micro_f1"] > best_val:
            best_val = val_metrics["micro_f1"]
            best_test = evaluate_masked(encoder, head, data, test_mask, device)
    return best_test


def run_gmnn_inspired(data, train_mask, val_mask, test_mask, device, epochs=100):
    in_dim = data.x.shape[1]
    num_labels = data.y.shape[1]
    encoder = GraphSAGEEncoder(in_dim, 256, 256).to(device)
    head = GMNNStyleHead(256, num_labels, n_rounds=2).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=0.005)

    x, edge_index, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)
    best_val, best_test = -1.0, None
    for epoch in range(1, epochs + 1):
        encoder.train(); head.train()
        optimizer.zero_grad()
        z = encoder(x, edge_index)
        logits = head(z, edge_index)
        loss = F.binary_cross_entropy_with_logits(logits[train_mask], y[train_mask])
        loss.backward(); optimizer.step()

        val_metrics = evaluate_masked(encoder, head, data, val_mask, device, needs_edge_index=True)
        if val_metrics["micro_f1"] > best_val:
            best_val = val_metrics["micro_f1"]
            best_test = evaluate_masked(encoder, head, data, test_mask, device, needs_edge_index=True)
    return best_test


def run_corgcn(data, train_mask, val_mask, test_mask, device, epochs=100):
    in_dim = data.x.shape[1]
    num_labels = data.y.shape[1]
    encoder = GraphSAGEEncoder(in_dim, 256, 256).to(device)
    head = CorGCNHead(256, num_labels, k_prime=20, proto_dim=64, top_lambda=5).to(device)
    group_map = build_macro_group_map(num_labels, head.k_prime, device)

    x, edge_index, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)
    macro_y_train = true_to_macro_targets(y[train_mask], group_map, head.k_prime)
    class_weights = 1.0 / torch.sqrt(macro_y_train.sum(dim=0) + 1e-9)
    focal_alpha = class_weights / class_weights.sum()
    focal_loss_fn = MultiLabelFocalLoss(alpha=focal_alpha)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=0.005)

    best_val, best_test = -1.0, None
    for epoch in range(1, epochs + 1):
        encoder.train(); head.train()
        optimizer.zero_grad()
        z = encoder(x, edge_index)
        logits, E_x = head(z, edge_index)

        macro_y = true_to_macro_targets(y, group_map, head.k_prime)
        L_cls = F.binary_cross_entropy_with_logits(logits[train_mask], y[train_mask])
        L_cmi = head.cmi_loss(E_x[train_mask], macro_y[train_mask])
        L_lm = head.lm_loss(E_x[train_mask], macro_y[train_mask], focal_loss_fn)
        alpha = (L_cls.detach() / (3 * L_cmi.detach() + 1e-8)).abs()
        beta = (L_cls.detach() / (3 * L_lm.detach() + 1e-8)).abs()
        loss = L_cls + alpha * L_cmi + beta * L_lm
        loss.backward(); optimizer.step()

        with torch.no_grad():
            encoder.eval(); head.eval()
            z_eval = encoder(x, edge_index)
            logits_eval, _ = head(z_eval, edge_index)
            val_probs = torch.sigmoid(logits_eval[val_mask]).cpu().numpy()
            val_metrics = compute_all_metrics(y[val_mask].cpu().numpy(), val_probs)
        if val_metrics["micro_f1"] > best_val:
            best_val = val_metrics["micro_f1"]
            test_probs = torch.sigmoid(logits_eval[test_mask]).cpu().numpy()
            best_test = compute_all_metrics(y[test_mask].cpu().numpy(), test_probs)
    return best_test


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    selected_indices = load_selected_labels()
    data = get_single_graph(selected_indices)
    n = data.num_nodes
    print(f"Using single graph: {n} nodes, {data.num_edges} edges, {data.y.shape[1]} labels")

    train_mask, val_mask, test_mask = make_transductive_masks(n)
    print(f"Split: {train_mask.sum().item()} train / {val_mask.sum().item()} val / {test_mask.sum().item()} test")

    print("\n=== Independent baseline (transductive) ===")
    baseline_test = run_independent(data, train_mask, val_mask, test_mask, device)
    print(format_metrics(baseline_test))

    print("\n=== GMNN-inspired (transductive) ===")
    gmnn_test = run_gmnn_inspired(data, train_mask, val_mask, test_mask, device)
    print(format_metrics(gmnn_test))

    print("\n=== CorGCN (transductive) ===")
    corgcn_test = run_corgcn(data, train_mask, val_mask, test_mask, device)
    print(format_metrics(corgcn_test))

    print("\n=== TRANSDUCTIVE SUMMARY ===")
    print(f"Baseline:      {baseline_test['micro_f1']:.4f}")
    print(f"GMNN-inspired: {gmnn_test['micro_f1']:.4f} (delta: {gmnn_test['micro_f1']-baseline_test['micro_f1']:+.4f})")
    print(f"CorGCN:        {corgcn_test['micro_f1']:.4f} (delta: {corgcn_test['micro_f1']-baseline_test['micro_f1']:+.4f})")
    print(f"\nCompare to INDUCTIVE results: baseline 0.8815, GMNN-inspired 0.9022 (+0.0207), CorGCN 0.8184-0.8199 (~-0.06)")

    os.makedirs("results", exist_ok=True)
    with open("results/transductive_test_results.json", "w") as f:
        json.dump(
            {
                "n_nodes": n,
                "baseline": baseline_test,
                "gmnn_inspired": gmnn_test,
                "corgcn": corgcn_test,
                "inductive_comparison": {
                    "baseline": 0.8815, "gmnn_inspired": 0.9022, "corgcn": 0.8199
                },
            },
            f,
            indent=2,
        )
    print("\nSaved to results/transductive_test_results.json")


if __name__ == "__main__":
    main()
