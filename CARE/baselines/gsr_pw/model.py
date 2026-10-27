"""Bi-GRU/attention generator used by the GSR-PW reproduction."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class TextEncoder(nn.Module):
    def __init__(self, embedding: nn.Embedding, word_dim: int, hidden_size: int, dropout: float):
        super().__init__()
        self.embedding = embedding
        self.gru = nn.GRU(
            word_dim,
            hidden_size // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(tokens))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, hidden = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=tokens.size(1)
        )
        final = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        return output, final


@dataclass
class EncodedContext:
    requirement: torch.Tensor
    requirement_mask: torch.Tensor
    category: torch.Tensor
    category_mask: torch.Tensor
    initial: torch.Tensor


class GSR(nn.Module):
    """GSR generator with current- and retrieved-requirement distributions."""

    def __init__(
        self,
        text_vocab_size: int,
        api_count: int,
        word_dim: int = 300,
        hidden_size: int = 200,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.api_count = api_count
        self.eos_id = api_count
        self.bos_id = api_count + 1
        self.output_size = api_count + 1
        self.hidden_size = hidden_size

        self.word_embedding = nn.Embedding(text_vocab_size, word_dim, padding_idx=0)
        self.requirement_encoder = TextEncoder(
            self.word_embedding, word_dim, hidden_size, dropout
        )
        self.category_encoder = TextEncoder(
            self.word_embedding, word_dim, hidden_size, dropout
        )
        self.service_embedding = nn.Embedding(api_count + 2, word_dim)
        self.decoder = nn.GRUCell(word_dim + hidden_size, hidden_size)
        self.output = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 3, self.output_size),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.word_embedding.weight, std=0.05)
        nn.init.normal_(self.service_embedding.weight, std=0.05)
        with torch.no_grad():
            self.word_embedding.weight[0].zero_()

    @staticmethod
    def _mask(tokens: torch.Tensor) -> torch.Tensor:
        return tokens.ne(0)

    def encode(
        self,
        requirements: torch.Tensor,
        requirement_lengths: torch.Tensor,
        categories: torch.Tensor,
        category_lengths: torch.Tensor,
    ) -> EncodedContext:
        requirement, initial = self.requirement_encoder(
            requirements, requirement_lengths
        )
        category, _ = self.category_encoder(categories, category_lengths)
        return EncodedContext(
            requirement=requirement,
            requirement_mask=self._mask(requirements),
            category=category,
            category_mask=self._mask(categories),
            initial=initial,
        )

    @staticmethod
    def _attention(
        sequence: torch.Tensor, mask: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        energy = torch.bmm(sequence, state.unsqueeze(-1)).squeeze(-1)
        energy = energy.masked_fill(~mask, -1e4)
        weights = torch.softmax(energy, dim=-1)
        return torch.bmm(weights.unsqueeze(1), sequence).squeeze(1)

    def _step(
        self,
        previous: torch.Tensor,
        state: torch.Tensor,
        context: EncodedContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.service_embedding(previous)
        state = self.decoder(torch.cat((embedded, context.initial), dim=-1), state)
        requirement = self._attention(
            context.requirement, context.requirement_mask, state
        )
        category = self._attention(context.category, context.category_mask, state)
        logits = self.output(torch.cat((state, requirement, category), dim=-1))
        return logits, state

    def _fused_log_probs(
        self,
        previous: torch.Tensor,
        current_state: torch.Tensor,
        retrieved_state: torch.Tensor,
        current: EncodedContext,
        retrieved: EncodedContext,
        similarity: torch.Tensor,
        gamma: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_logits, current_state = self._step(previous, current_state, current)
        retrieved_logits, retrieved_state = self._step(
            previous, retrieved_state, retrieved
        )
        current_prob = torch.softmax(current_logits, dim=-1)
        retrieved_prob = torch.softmax(retrieved_logits, dim=-1)
        weight = (gamma * similarity).clamp_min(0.0).unsqueeze(-1)
        fused = (current_prob + weight * retrieved_prob) / (1.0 + weight)
        return fused.clamp_min(1e-9).log(), current_state, retrieved_state

    def teacher_forced_nll(
        self,
        current: EncodedContext,
        retrieved: EncodedContext,
        similarity: torch.Tensor,
        targets: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        batch = targets.size(0)
        previous = torch.full(
            (batch,), self.bos_id, dtype=torch.long, device=targets.device
        )
        current_state = current.initial
        retrieved_state = retrieved.initial
        losses = []
        for position in range(targets.size(1)):
            log_prob, current_state, retrieved_state = self._fused_log_probs(
                previous,
                current_state,
                retrieved_state,
                current,
                retrieved,
                similarity,
                gamma,
            )
            target = targets[:, position]
            valid = target.ge(0)
            safe_target = target.masked_fill(~valid, 0)
            losses.append(-log_prob.gather(1, safe_target[:, None]).squeeze(1) * valid)
            previous = torch.where(valid, safe_target, previous)
        loss = torch.stack(losses, dim=1)
        denominator = targets.ge(0).sum(dim=1).clamp_min(1)
        return (loss.sum(dim=1) / denominator).mean()

    def _rollout(
        self,
        current: EncodedContext,
        retrieved: EncodedContext,
        similarity: torch.Tensor,
        gamma: float,
        max_steps: int,
        sample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = similarity.size(0)
        device = similarity.device
        previous = torch.full((batch,), self.bos_id, dtype=torch.long, device=device)
        current_state = current.initial
        retrieved_state = retrieved.initial
        active = torch.ones(batch, dtype=torch.bool, device=device)
        used = torch.zeros(batch, self.api_count, dtype=torch.bool, device=device)
        sequences, chosen_log_probs, active_masks = [], [], []
        for _ in range(max_steps):
            log_prob, current_state, retrieved_state = self._fused_log_probs(
                previous,
                current_state,
                retrieved_state,
                current,
                retrieved,
                similarity,
                gamma,
            )
            masked = log_prob.clone()
            masked[:, : self.api_count] = masked[:, : self.api_count].masked_fill(
                used, -1e9
            )
            if sample:
                chosen = torch.distributions.Categorical(logits=masked).sample()
            else:
                chosen = masked.argmax(dim=-1)
            chosen = torch.where(
                active, chosen, torch.full_like(chosen, self.eos_id)
            )
            sequences.append(chosen)
            active_masks.append(active)
            chosen_log_probs.append(
                log_prob.gather(1, chosen[:, None]).squeeze(1) * active
            )
            selected_api = active & chosen.lt(self.api_count)
            if selected_api.any():
                used[selected_api, chosen[selected_api]] = True
            active = active & chosen.ne(self.eos_id)
            previous = chosen
        return (
            torch.stack(sequences, dim=1),
            torch.stack(chosen_log_probs, dim=1),
            torch.stack(active_masks, dim=1),
        )

    def scst_loss(
        self,
        current: EncodedContext,
        retrieved: EncodedContext,
        similarity: torch.Tensor,
        target_sets: torch.Tensor,
        gamma: float,
        max_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sampled, sampled_log_prob, sampled_mask = self._rollout(
            current, retrieved, similarity, gamma, max_steps, sample=True
        )
        with torch.no_grad():
            greedy, _, _ = self._rollout(
                current, retrieved, similarity, gamma, max_steps, sample=False
            )
            sample_reward = self._set_reward(sampled, target_sets)
            greedy_reward = self._set_reward(greedy, target_sets)
            advantage = sample_reward - greedy_reward
        lengths = sampled_mask.sum(dim=1).clamp_min(1)
        log_likelihood = sampled_log_prob.sum(dim=1) / lengths
        return -(advantage * log_likelihood).mean(), sample_reward.mean(), greedy_reward.mean()

    def _set_reward(self, generated: torch.Tensor, target_sets: torch.Tensor) -> torch.Tensor:
        rewards = []
        for generated_row, target_row in zip(generated, target_sets):
            predicted = {
                int(value)
                for value in generated_row.tolist()
                if 0 <= int(value) < self.api_count
            }
            target = {int(value) for value in target_row.tolist() if int(value) >= 0}
            hits = len(predicted & target)
            precision = hits / max(1, len(predicted))
            recall = hits / max(1, len(target))
            rewards.append(precision + recall)
        return torch.tensor(rewards, dtype=torch.float32, device=generated.device)

    @torch.no_grad()
    def first_step_scores(
        self,
        current: EncodedContext,
        retrieved: EncodedContext,
        similarity: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        previous = torch.full(
            (similarity.size(0),),
            self.bos_id,
            dtype=torch.long,
            device=similarity.device,
        )
        log_prob, _, _ = self._fused_log_probs(
            previous,
            current.initial,
            retrieved.initial,
            current,
            retrieved,
            similarity,
            gamma,
        )
        return log_prob[:, : self.api_count]

