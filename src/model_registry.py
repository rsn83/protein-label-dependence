"""
Model registry: factory functions for encoders and heads, decoupled from
any specific dataset.

Encoders: "gcn", "sage"
Heads: "independent", "dependency", "mlgcn_chen", "mlgcn_gao", "gmnn", "lamp", "corgcn"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))
from train_baseline import GraphSAGEEncoder
from train_corgcn_precise import (
    GCNEncoder, CorGCNHead, MultiLabelFocalLoss,
    build_macro_group_map, true_to_macro_targets
)
from train_dependency_baseline import DependencyCorrectionHead
from train_gmnn_style_baseline import GMNNStyleHead
from train_lamp_style_baseline import LaMPHead
from train_mlgcn_style_baseline import MLGCNHead, build_reweighted_adjacency
from train_mlgcn_gao_style import MLGCNGaoHead, compute_unigram_noise_dist


ENCODERS = {
    "gcn": GCNEncoder,
    "sage": GraphSAGEEncoder,
}


def get_encoder(encoder_type: str, in_dim: int, hidden_dim: int, device):
    if encoder_type not in ENCODERS:
        raise ValueError(f"Unknown encoder_type: {encoder_type}. Choose from {list(ENCODERS)}")
    return ENCODERS[encoder_type](in_dim, hidden_dim, hidden_dim).to(device)


class IndependentHead(nn.Module):
    def __init__(self, embed_dim, num_labels):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_labels)
        self.needs_edge_index = False

    def forward(self, z, edge_index=None):
        return self.linear(z)


def build_generic_dependency_edges(y_train: np.ndarray, tau: float = 0.4, max_edges: int = 200):
    n_labels = y_train.shape[1]
    co_occur = y_train.T @ y_train
    label_counts = y_train.sum(axis=0)
    cond_prob = co_occur / np.maximum(label_counts[:, None], 1)
    np.fill_diagonal(cond_prob, 0)

    edges = []
    for i in range(n_labels):
        for j in range(n_labels):
            if i != j and cond_prob[i, j] >= tau:
                edges.append((i, j))
    if len(edges) > max_edges:
        edges = edges[:max_edges]
    if len(edges) == 0:
        for i in range(n_labels):
            j = np.argmax(cond_prob[i])
            if i != j:
                edges.append((i, int(j)))
    return edges


HEAD_REQUIRES_EDGE_INDEX = {
    "independent": False,
    "dependency": False,
    "mlgcn_chen": False,
    "mlgcn_gao": False,
    "gmnn": True,
    "lamp": False,
    "corgcn": True,
}


def get_head(head_type: str, embed_dim: int, num_labels: int, device, dependency_edges=None, y_train_np=None):
    if head_type == "independent":
        return IndependentHead(embed_dim, num_labels).to(device), {}

    elif head_type == "dependency":
        if dependency_edges is None:
            raise ValueError("dependency head requires dependency_edges (see build_generic_dependency_edges)")
        return DependencyCorrectionHead(embed_dim, num_labels, dependency_edges).to(device), {}

    elif head_type == "mlgcn_chen":
        if y_train_np is None:
            raise ValueError("mlgcn_chen head requires y_train_np to build its co-occurrence adjacency")
        adjacency = build_reweighted_adjacency(y_train_np, tau=0.4, p=0.2)
        return MLGCNHead(embed_dim, num_labels, adjacency).to(device), {}

    elif head_type == "mlgcn_gao":
        if y_train_np is None:
            raise ValueError("mlgcn_gao head requires y_train_np to build its unigram noise distribution")
        head = MLGCNGaoHead(embed_dim, num_labels).to(device)
        noise_dist = compute_unigram_noise_dist(y_train_np, device)
        return head, {"noise_dist": noise_dist}

    elif head_type == "gmnn":
        return GMNNStyleHead(embed_dim, num_labels, n_rounds=2).to(device), {}

    elif head_type == "lamp":
        return LaMPHead(embed_dim, num_labels, label_dim=128, n_heads=4, n_rounds=2).to(device), {}

    elif head_type == "corgcn":
        k_prime = min(20, num_labels)
        head = CorGCNHead(embed_dim, num_labels, k_prime=k_prime, proto_dim=64, top_lambda=7).to(device)
        group_map = build_macro_group_map(num_labels, k_prime, device)
        return head, {"group_map": group_map, "k_prime": k_prime}

    else:
        raise ValueError(f"Unknown head_type: {head_type}. Choose from {list(HEAD_REQUIRES_EDGE_INDEX)}")


HEAD_TYPES = list(HEAD_REQUIRES_EDGE_INDEX.keys())
ENCODER_TYPES = list(ENCODERS.keys())
