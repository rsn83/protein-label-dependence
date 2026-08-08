"""
ML-GCN (Gao, Zhang, Zhou — WISE 2019), "Semi-Supervised Graph Embedding for
Multi-Label Graph Node Classification". Distinct from Chen et al. (CVPR 2019)
"mlgcn_chen" already in this pipeline — different mechanism entirely.

Mechanism (Eq 1-9 in the paper):
  - Standard GCN encoder (here: reuses whichever encoder — gcn/sage — is
    already producing z, matching this pipeline's decoupled design).
  - Trainable label embedding matrix Z_Y (same dim as z), NOT fixed.
  - Node-label Skip-gram loss (Eq 5,7): pulls a node's embedding close to
    its own labels' embeddings, negative sampling with unigram^0.75 noise.
  - Label-label Skip-gram loss (Eq 6,8): pulls co-occurring labels' embeddings
    close to each other, same negative sampling scheme.
  - Combined: L = lambda1*L_label-label + lambda2*L_node-label + L_sigmoid
    (paper uses lambda1=lambda2=0.25 fixed).

Deliberate efficiency simplification (stated): label-label pairs are CAPPED
per node (max_pairs_per_node) rather than enumerating all C(k,2) ordered
pairs — necessary for tractability on datasets with many labels per node
(e.g. SagePPI, up to 80 labels), avoiding a combinatorial blowup per epoch.
"""

import torch
import torch.nn as nn
import numpy as np


class MLGCNGaoHead(nn.Module):
    def __init__(self, embed_dim, num_labels, K=5, lambda1=0.25, lambda2=0.25, max_pairs_per_node=10):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_labels)
        self.label_emb = nn.Parameter(torch.randn(num_labels, embed_dim) * 0.1)
        self.embed_dim = embed_dim
        self.num_labels = num_labels
        self.K = K
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_pairs_per_node = max_pairs_per_node

    def forward(self, z):
        return self.classifier(z)

    def node_label_loss(self, z, y, noise_dist, device):
        pos_node_idx, pos_label_idx = y.nonzero(as_tuple=True)
        if len(pos_node_idx) == 0:
            return torch.tensor(0.0, device=device)

        h = z[pos_node_idx]
        zy_pos = self.label_emb[pos_label_idx]
        pos_score = torch.sigmoid((h * zy_pos).sum(dim=1))
        pos_loss = -torch.log(pos_score + 1e-10)

        neg_idx = torch.multinomial(noise_dist, len(pos_node_idx) * self.K, replacement=True)
        neg_idx = neg_idx.view(len(pos_node_idx), self.K)
        zy_neg = self.label_emb[neg_idx]
        neg_score = torch.sigmoid(-(h.unsqueeze(1) * zy_neg).sum(dim=2))
        neg_loss = -torch.log(neg_score + 1e-10).sum(dim=1)

        return (pos_loss + neg_loss).mean()

    def label_label_loss(self, y, noise_dist, device):
        y_cpu = y.cpu().numpy()
        all_i, all_j = [], []
        for row in range(y_cpu.shape[0]):
            pos = np.nonzero(y_cpu[row])[0]
            if len(pos) < 2:
                continue
            n_pairs_available = len(pos) * (len(pos) - 1)
            if n_pairs_available <= self.max_pairs_per_node:
                for a in pos:
                    for b in pos:
                        if a != b:
                            all_i.append(a)
                            all_j.append(b)
            else:
                idx_pairs = np.random.choice(len(pos), size=(self.max_pairs_per_node, 2))
                for a, b in idx_pairs:
                    if a != b:
                        all_i.append(pos[a])
                        all_j.append(pos[b])

        if len(all_i) == 0:
            return torch.tensor(0.0, device=device)

        yi_idx = torch.tensor(all_i, device=device, dtype=torch.long)
        yj_idx = torch.tensor(all_j, device=device, dtype=torch.long)

        zy_i = self.label_emb[yi_idx]
        zy_j = self.label_emb[yj_idx]
        pos_score = torch.sigmoid((zy_i * zy_j).sum(dim=1))
        pos_loss = -torch.log(pos_score + 1e-10)

        neg_idx = torch.multinomial(noise_dist, len(all_i) * self.K, replacement=True)
        neg_idx = neg_idx.view(len(all_i), self.K)
        zy_neg = self.label_emb[neg_idx]
        neg_score = torch.sigmoid(-(zy_i.unsqueeze(1) * zy_neg).sum(dim=2))
        neg_loss = -torch.log(neg_score + 1e-10).sum(dim=1)

        return (pos_loss + neg_loss).mean()


def compute_unigram_noise_dist(y_train_np, device):
    counts = y_train_np.sum(axis=0) + 1e-9
    noise = counts ** 0.75
    noise = noise / noise.sum()
    return torch.tensor(noise, device=device, dtype=torch.float32)
