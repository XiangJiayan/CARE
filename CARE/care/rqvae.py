from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RQVAEConfig:
    input_dim: int
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    latent_dim: int = 32
    num_levels: int = 4
    codebook_size: int = 512
    beta: float = 0.25
    quant_weight: float = 1.0
    dropout: float = 0.0

    def to_dict(self) -> dict:
        result = asdict(self)
        result["hidden_dims"] = list(self.hidden_dims)
        return result

    @classmethod
    def from_dict(cls, values: dict) -> "RQVAEConfig":
        values = dict(values)
        values["hidden_dims"] = tuple(values["hidden_dims"])
        return cls(**values)


def _mlp(dimensions: Sequence[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (in_dim, out_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        if dropout:
            layers.append(nn.Dropout(dropout))
        linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_normal_(linear.weight)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if index < len(dimensions) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


@torch.no_grad()
def torch_kmeans(samples: torch.Tensor, clusters: int, iterations: int = 50) -> torch.Tensor:
    if len(samples) < clusters:
        raise ValueError(f"Need at least {clusters} samples for codebook initialization")
    indices = torch.linspace(0, len(samples) - 1, clusters, device=samples.device).long()
    centers = samples[indices].clone()
    for _ in range(iterations):
        assignments = torch.cdist(samples, centers).argmin(dim=1)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, assignments, samples)
        counts = torch.bincount(assignments, minlength=clusters)
        new_centers = torch.where(counts[:, None] > 0, sums / counts.clamp_min(1)[:, None], centers)
        if torch.allclose(new_centers, centers, atol=1e-5, rtol=1e-4):
            break
        centers = new_centers
    return centers


class VectorQuantizer(nn.Module):
    def __init__(self, size: int, dimension: int, beta: float) -> None:
        super().__init__()
        self.beta = beta
        self.embedding = nn.Embedding(size, dimension)
        nn.init.uniform_(self.embedding.weight, -1.0 / size, 1.0 / size)
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def initialize(self, samples: torch.Tensor) -> None:
        self.embedding.weight.copy_(torch_kmeans(samples, self.embedding.num_embeddings))
        self.initialized.fill_(True)

    def forward(self, inputs: torch.Tensor, initialize: bool = False):
        flat = inputs.reshape(-1, inputs.shape[-1])
        if initialize and not bool(self.initialized.item()):
            self.initialize(flat.detach())
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1).unsqueeze(0)
            - 2 * flat @ self.embedding.weight.t()
        )
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view_as(inputs)
        codebook_loss = F.mse_loss(quantized, inputs.detach())
        commitment_loss = F.mse_loss(quantized.detach(), inputs)
        loss = codebook_loss + self.beta * commitment_loss
        straight_through = inputs + (quantized - inputs).detach()
        return straight_through, quantized, loss, indices.view(inputs.shape[:-1])


class ResidualQuantizer(nn.Module):
    def __init__(self, levels: int, codebook_size: int, dimension: int, beta: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantizer(codebook_size, dimension, beta) for _ in range(levels)]
        )

    def forward(self, inputs: torch.Tensor, initialize: bool = False):
        residual = inputs
        quantized_sum = torch.zeros_like(inputs)
        losses = []
        codes = []
        for layer in self.layers:
            _, quantized, loss, indices = layer(residual, initialize=initialize)
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized.detach()
            losses.append(loss)
            codes.append(indices)
        # A single STE gives the encoder identity gradient, not num_levels times it.
        straight_through = inputs + (quantized_sum - inputs).detach()
        return straight_through, torch.stack(losses).sum(), torch.stack(codes, dim=-1)


class RQVAE(nn.Module):
    def __init__(self, config: RQVAEConfig) -> None:
        super().__init__()
        self.config = config
        encoder_dims = [config.input_dim, *config.hidden_dims, config.latent_dim]
        decoder_dims = list(reversed(encoder_dims))
        self.encoder = _mlp(encoder_dims, config.dropout)
        self.quantizer = ResidualQuantizer(
            config.num_levels, config.codebook_size, config.latent_dim, config.beta
        )
        self.decoder = _mlp(decoder_dims, config.dropout)

    def forward(self, inputs: torch.Tensor, initialize: bool = False):
        latent = self.encoder(inputs)
        quantized, quantization_loss, codes = self.quantizer(latent, initialize=initialize)
        reconstructed = self.decoder(quantized)
        reconstruction_loss = F.mse_loss(reconstructed, inputs)
        total_loss = reconstruction_loss + self.config.quant_weight * quantization_loss
        return total_loss, reconstruction_loss, quantization_loss, codes

    @torch.no_grad()
    def encode_codes(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(inputs)
        _, _, codes = self.quantizer(latent, initialize=False)
        return codes
