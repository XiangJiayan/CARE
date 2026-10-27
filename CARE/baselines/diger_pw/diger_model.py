"""Differentiable residual semantic tokenizer used by DIGER-PW.

This implements the corrected DIGER principle: deterministic hard semantic IDs
are used in the forward pass, while current soft assignments carry gradients
back to the tokenizer. FrqUD applies Gumbel exploration to over-used codes only.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class AssignmentOutput:
    probabilities: torch.Tensor
    hard_codes: torch.Tensor
    reconstruction_loss: torch.Tensor
    quantization_loss: torch.Tensor


class DifferentiableResidualTokenizer(nn.Module):
    def __init__(self, rqvae: nn.Module, temperature: float = 2.0,
                 frequency_ratio: float = 1.5, usage_ema: float = 0.99) -> None:
        super().__init__()
        self.rqvae = rqvae
        self.temperature = float(temperature)
        self.frequency_ratio = float(frequency_ratio)
        self.usage_ema = float(usage_ema)
        levels = rqvae.config.num_levels
        size = rqvae.config.codebook_size
        self.register_buffer("usage", torch.full((levels, size), 1.0 / size))

    @property
    def levels(self) -> int:
        return int(self.rqvae.config.num_levels)

    @property
    def codebook_size(self) -> int:
        return int(self.rqvae.config.codebook_size)

    @staticmethod
    def _distance_logits(values: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        distances = (
            values.pow(2).sum(-1, keepdim=True)
            + codebook.pow(2).sum(-1).unsqueeze(0)
            - 2.0 * values @ codebook.t()
        )
        return -distances

    def initialize_usage(self, vectors: torch.Tensor) -> None:
        with torch.no_grad():
            codes = self.deterministic_codes(vectors)
            rows = []
            for level in range(self.levels):
                count = torch.bincount(codes[:, level], minlength=self.codebook_size).float()
                rows.append(count / count.sum().clamp_min(1.0))
            self.usage.copy_(torch.stack(rows))

    @torch.no_grad()
    def update_usage(self, hard_codes: torch.Tensor) -> None:
        for level in range(self.levels):
            count = torch.bincount(hard_codes[:, level], minlength=self.codebook_size).float()
            frequency = count / count.sum().clamp_min(1.0)
            self.usage[level].mul_(self.usage_ema).add_(frequency * (1.0 - self.usage_ema))

    @torch.no_grad()
    def deterministic_codes(self, vectors: torch.Tensor) -> torch.Tensor:
        latent = self.rqvae.encoder(vectors)
        residual = latent
        codes = []
        for layer in self.rqvae.quantizer.layers:
            logits = self._distance_logits(residual, layer.embedding.weight)
            indices = logits.argmax(-1)
            quantized = layer.embedding(indices)
            residual = residual - quantized
            codes.append(indices)
        return torch.stack(codes, dim=-1)

    @torch.no_grad()
    def sinkhorn_codes(self, vectors: torch.Tensor, epsilon: float = 0.003,
                       iterations: int = 50) -> torch.Tensor:
        """Build balanced full-catalog IDs exactly as DIGER evaluation does."""
        latent = self.rqvae.encoder(vectors)
        residual = latent
        codes = []
        for layer in self.rqvae.quantizer.layers:
            weight = layer.embedding.weight
            distances = -self._distance_logits(residual, weight)
            maximum, minimum = distances.max(), distances.min()
            middle = (maximum + minimum) / 2.0
            amplitude = (maximum - middle).clamp_min(1e-5)
            centered = ((distances - middle) / amplitude).double()
            assignment = torch.exp(-centered / float(epsilon))
            assignment /= assignment.sum().clamp_min(1e-300)
            batch, codebook = assignment.shape
            for _ in range(int(iterations)):
                assignment /= assignment.sum(dim=1, keepdim=True).clamp_min(1e-300)
                assignment /= batch
                assignment /= assignment.sum(dim=0, keepdim=True).clamp_min(1e-300)
                assignment /= codebook
            assignment *= batch
            indices = assignment.argmax(-1)
            quantized = layer.embedding(indices)
            residual = residual - quantized
            codes.append(indices)
        return torch.stack(codes, dim=-1)

    def forward(self, vectors: torch.Tensor, add_noise: bool = True,
                cached_hard_codes: torch.Tensor | None = None) -> AssignmentOutput:
        latent = self.rqvae.encoder(vectors)
        residual = latent
        quantized_sum = torch.zeros_like(latent)
        probabilities, hard_codes, vq_losses = [], [], []
        for level, layer in enumerate(self.rqvae.quantizer.layers):
            logits = self._distance_logits(residual, layer.embedding.weight)
            deterministic = logits.argmax(-1)
            noisy_logits = logits
            if self.training and add_noise:
                with torch.no_grad():
                    count = torch.bincount(deterministic, minlength=self.codebook_size).float()
                    frequency = count / count.sum().clamp_min(1.0)
                    self.usage[level].mul_(self.usage_ema).add_(
                        frequency * (1.0 - self.usage_ema)
                    )
                # Official FrqUD decides per assignment: an item receives
                # Gumbel exploration when its deterministic code is over-used.
                hot_assignment = (
                    self.usage[level][deterministic]
                    > (self.frequency_ratio / self.codebook_size)
                )
                uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
                gumbel = -torch.log(-torch.log(uniform))
                noisy_logits = logits + gumbel * hot_assignment.to(logits.dtype).unsqueeze(-1)
            soft = F.softmax(noisy_logits / self.temperature, dim=-1)
            current_indices = noisy_logits.argmax(-1)
            current_hard = F.one_hot(current_indices, self.codebook_size).to(logits.dtype)
            current_st = current_hard - soft.detach() + soft

            # The tokenizer's own reconstruction/VQ path uses its current hard
            # assignment, exactly as the corrected DIGER quantizer does.
            quantized = current_st @ layer.embedding.weight

            # The recommender sees cached pretrained hard IDs in forward, while
            # gradients follow the tokenizer's current soft assignments.
            recommender_codes = (current_indices if cached_hard_codes is None
                                 else cached_hard_codes[:, level])
            recommender_hard = F.one_hot(
                recommender_codes, self.codebook_size
            ).to(logits.dtype)
            recommender_st = recommender_hard - soft.detach() + soft

            codebook_loss = F.mse_loss(quantized, residual.detach())
            commitment_loss = F.mse_loss(quantized.detach(), residual)
            vq_losses.append(codebook_loss + layer.beta * commitment_loss)
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized
            probabilities.append(recommender_st)
            hard_codes.append(deterministic)

        reconstructed = self.rqvae.decoder(quantized_sum)
        reconstruction = F.mse_loss(reconstructed, vectors)
        return AssignmentOutput(
            probabilities=torch.stack(probabilities, dim=1),
            hard_codes=torch.stack(hard_codes, dim=1),
            reconstruction_loss=reconstruction,
            quantization_loss=torch.stack(vq_losses).sum(),
        )


def differentiable_generation_loss(generator, query_vectors: torch.Tensor,
                                   assignments: torch.Tensor,
                                   suffix_ids: torch.Tensor,
                                   starts: list[int]) -> torch.Tensor:
    """Autoregressive SID loss with soft-backward semantic targets."""
    encoded, attention_mask = generator.encode(query_vectors)
    batch, levels, codebook_size = assignments.shape
    shared = generator.t5.shared.weight

    start_embedding = shared.new_zeros((batch, shared.shape[1]))
    start_embedding.copy_(shared[generator.t5.config.decoder_start_token_id])
    decoder_embeddings = [start_embedding]
    for level in range(levels):
        token_embeddings = shared[starts[level]: starts[level] + codebook_size]
        decoder_embeddings.append(assignments[:, level] @ token_embeddings)
    suffix_tokens = suffix_ids + starts[levels]
    decoder_embeddings.append(shared[suffix_tokens])
    decoder_inputs = torch.stack(decoder_embeddings, dim=1)

    logits = generator.t5(
        encoder_outputs=encoded,
        attention_mask=attention_mask,
        decoder_inputs_embeds=decoder_inputs,
        use_cache=False,
        return_dict=True,
    ).logits.float()
    losses = []
    for level in range(levels):
        local = logits[:, level, starts[level]: starts[level] + codebook_size]
        losses.append(-(assignments[:, level] * F.log_softmax(local, -1)).sum(-1))
    losses.append(F.cross_entropy(logits[:, levels], suffix_tokens, reduction="none"))
    eos = torch.ones(batch, dtype=torch.long, device=logits.device)
    losses.append(F.cross_entropy(logits[:, levels + 1], eos, reduction="none"))
    return torch.stack(losses, dim=1).mean()
