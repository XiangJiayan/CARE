"""TSSGCF model following IntelligentServiceLab/TSSGCF."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TSSGCF(nn.Module):
    def __init__(self, mashups: int, apis: int, text_dim: int, dim: int,
                 layers: int, text_weight: float):
        super().__init__()
        self.layers = layers
        self.text_weight = text_weight
        self.mashup_embedding = nn.Embedding(mashups, dim)
        self.api_embedding = nn.Embedding(apis, dim)
        self.text_mlp = nn.Linear(text_dim, dim)
        nn.init.normal_(self.mashup_embedding.weight, std=0.01)
        nn.init.normal_(self.api_embedding.weight, std=0.01)

    @staticmethod
    def graph_encode(adjacency: torch.Tensor, embeddings: torch.Tensor,
                     layers: int) -> torch.Tensor:
        values = embeddings
        outputs = [values]
        for _ in range(layers):
            values = torch.sparse.mm(adjacency, values)
            outputs.append(values)
        return F.normalize(torch.stack(outputs).mean(dim=0), p=2, dim=1)

    def forward(self, mashup_adj: torch.Tensor, api_adj: torch.Tensor,
                mashup_text: torch.Tensor, api_text: torch.Tensor):
        graph_m = self.graph_encode(mashup_adj, self.mashup_embedding.weight, self.layers)
        graph_a = self.graph_encode(api_adj, self.api_embedding.weight, self.layers)
        text_m = F.normalize(self.text_mlp(mashup_text), p=2, dim=1)
        text_a = F.normalize(self.text_mlp(api_text), p=2, dim=1)
        final_m = (1.0 - self.text_weight) * graph_m + self.text_weight * text_m
        final_a = (1.0 - self.text_weight) * graph_a + self.text_weight * text_a
        return graph_m, graph_a, final_m, final_a


def textual_similarity_loss(graph: torch.Tensor, text: torch.Tensor,
                            indices: torch.Tensor, threshold: float,
                            beta: float) -> torch.Tensor:
    """Penalize positive graph similarity between textually dissimilar nodes."""
    g = F.normalize(graph[indices], p=2, dim=1)
    t = F.normalize(text[indices], p=2, dim=1)
    text_sim = t @ t.T
    graph_sim = g @ g.T
    eye = torch.eye(len(indices), dtype=torch.bool, device=indices.device)
    mask = (text_sim < threshold) & ~eye
    if not mask.any():
        return graph_sim.sum() * 0.0
    text_gap = F.relu(threshold - text_sim[mask]).pow(beta)
    graph_penalty = F.relu(graph_sim[mask]).pow(beta + 1.0)
    return (text_gap * graph_penalty).mean()

