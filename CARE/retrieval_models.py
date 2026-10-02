"""Content retriever and score-fusion utilities for CARE."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ResidualTower(nn.Module):
    """Low-rank adapter initialized as an identity mapping."""
    def __init__(self, dimension=768, bottleneck=32):
        super().__init__()
        self.down = nn.Linear(dimension, bottleneck)
        self.up = nn.Linear(bottleneck, dimension)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        delta = self.up(F.gelu(self.down(x)))
        return F.normalize(x + delta, dim=-1), delta


class SemanticRetriever(nn.Module):
    def __init__(self, dimension=768, bottleneck=32, temperature=0.05):
        super().__init__()
        self.query_tower = ResidualTower(dimension, bottleneck)
        self.api_tower = ResidualTower(dimension, bottleneck)
        self.log_temperature = nn.Parameter(torch.tensor(float(np.log(temperature))))

    def encode_query(self, x):
        return self.query_tower(x)

    def encode_api(self, x):
        return self.api_tower(x)

    def loss(self, query, api, positive, target_weight=None, anchor_weight=0.001):
        q, q_delta = self.encode_query(query)
        a, a_delta = self.encode_api(api)
        temperature = self.log_temperature.exp().clamp(0.01, 0.20)
        log_probability = F.log_softmax(q @ a.T / temperature, dim=-1)
        weight = positive.float()
        if target_weight is not None:
            weight = weight * target_weight[None, :]
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
        alignment = -(weight * log_probability).sum(-1).mean()
        anchor = q_delta.pow(2).mean() + a_delta.pow(2).mean()
        return alignment + anchor_weight * anchor, alignment, anchor, temperature


def row_standardize(score):
    score = np.asarray(score, dtype=np.float32)
    mean = score.mean(axis=1, keepdims=True)
    std = score.std(axis=1, keepdims=True)
    return (score - mean) / np.maximum(std, 1e-6)


def fused_score(generator_score, retrieval_score, alpha, popularity=None, beta=0.0):
    generator = row_standardize(generator_score)
    retrieval = row_standardize(retrieval_score)
    alpha = np.asarray(alpha, dtype=np.float32)
    if alpha.ndim == 0:
        alpha = np.full((len(generator), 1), float(alpha), dtype=np.float32)
    elif alpha.ndim == 1:
        alpha = alpha[:, None]
    result = alpha * generator + (1.0 - alpha) * retrieval
    if popularity is not None and beta:
        prior = np.asarray(popularity, dtype=np.float32)
        prior = (prior - prior.mean()) / max(float(prior.std()), 1e-6)
        result = result - float(beta) * prior[None, :]
    return result
