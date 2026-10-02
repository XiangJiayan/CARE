"""MPGCF model adapted from IntelligentServiceLab/MPGCF.

The mathematical components follow the official implementation: accuracy and
non-accuracy graph encoders, structure/text contrastive alignment, semantic
fusion, and head/tail popularity-smoothed BPR.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MPGCF(nn.Module):
    def __init__(self, users: int, items: int, text_dim: int, dim: int,
                 layers: int, alpha: float, text_weight: float):
        super().__init__()
        self.users = users
        self.items = items
        self.layers = layers
        self.alpha = alpha
        self.text_weight = text_weight
        self.user_embedding = nn.Embedding(users, dim)
        self.item_embedding = nn.Embedding(items, dim)
        self.text_projector = nn.Sequential(nn.Linear(text_dim, dim), nn.ReLU())
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def propagate(self, acc_adj: torch.Tensor, nacc_adj: torch.Tensor):
        ego = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        acc, nacc = ego, ego
        acc_layers, nacc_layers = [ego], [ego]
        for _ in range(self.layers):
            acc = torch.sparse.mm(acc_adj, acc)
            nacc = torch.sparse.mm(nacc_adj, nacc)
            acc_layers.append(acc)
            nacc_layers.append(nacc)
        acc_all = torch.stack(acc_layers).mean(dim=0)
        nacc_all = torch.stack(nacc_layers).mean(dim=0)
        acc_user, acc_item = torch.split(acc_all, [self.users, self.items])
        nacc_user, nacc_item = torch.split(nacc_all, [self.users, self.items])
        user = self.alpha * acc_user + (1.0 - self.alpha) * nacc_user
        item = self.alpha * acc_item + (1.0 - self.alpha) * nacc_item
        return acc_user, acc_item, nacc_user, nacc_item, user, item

    def project_text(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.text_projector(vectors)

    def fuse(self, graph: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        text = F.normalize(text, p=2, dim=-1)
        return (1.0 - self.text_weight) * graph + self.text_weight * text


def indexed_info_nce(anchor: torch.Tensor, positive: torch.Tensor,
                     all_targets: torch.Tensor, labels: torch.Tensor,
                     temperature: float) -> torch.Tensor:
    anchor = F.normalize(anchor, p=2, dim=-1)
    positive = F.normalize(positive, p=2, dim=-1)
    all_targets = F.normalize(all_targets, p=2, dim=-1)
    positive_score = (anchor * positive).sum(dim=-1) / temperature
    logits = anchor @ all_targets.T / temperature
    return -(positive_score - torch.logsumexp(logits, dim=1)).mean()


def symmetric_indexed_info_nce(left: torch.Tensor, right: torch.Tensor,
                               all_left: torch.Tensor, all_right: torch.Tensor,
                               labels: torch.Tensor, temperature: float) -> torch.Tensor:
    return 0.5 * (
        indexed_info_nce(left, right, all_right, labels, temperature)
        + indexed_info_nce(right, left, all_left, labels, temperature)
    )

