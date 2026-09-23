# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exportable DFlash sampling.

The two modules below are exported as CUDA DFlash methods so draft sampling,
speculative verification, and correction sampling run on the device. They
mirror the host sampler in ``runtime/engine/sampling.h``: stable top-k,
top-p over the sorted distribution, inverse-CDF sampling in original token
order, and exact argmax at temperature 0. Uniforms are drawn in-graph; the
pure functions take them explicitly so tests can pin them.
"""

from __future__ import annotations

import torch
from torch import nn


def sampling_probabilities(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``[R, V]`` filtered probabilities and ``[R]`` argmax tokens."""
    vocab_size = logits.shape[-1]
    logits = logits.float()
    greedy_tokens = torch.argmax(logits, dim=-1)

    safe_temperature = torch.where(
        temperature > 0.0, temperature, torch.ones_like(temperature)
    )
    sorted_logits, sorted_indices = torch.sort(
        logits / safe_temperature, dim=-1, descending=True, stable=True
    )
    ranks = torch.arange(vocab_size, device=logits.device)
    effective_top_k = torch.where(
        (top_k > 0) & (top_k < vocab_size), top_k, torch.full_like(top_k, vocab_size)
    )
    sorted_logits = torch.where(
        ranks < effective_top_k,
        sorted_logits,
        torch.full_like(sorted_logits, float("-inf")),
    )
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)

    mass_before_token = (
        torch.cumsum(sorted_probabilities, dim=-1) - sorted_probabilities
    )
    top_p_enabled = (top_p > 0.0) & (top_p < 1.0)
    sorted_probabilities = torch.where(
        ~top_p_enabled | (mass_before_token < top_p),
        sorted_probabilities,
        torch.zeros_like(sorted_probabilities),
    )
    sorted_probabilities = sorted_probabilities / sorted_probabilities.sum(
        dim=-1, keepdim=True
    )
    probabilities = torch.zeros_like(sorted_probabilities).scatter(
        -1, sorted_indices, sorted_probabilities
    )

    greedy_probabilities = torch.nn.functional.one_hot(
        greedy_tokens, num_classes=vocab_size
    ).to(probabilities.dtype)
    probabilities = torch.where(temperature > 0.0, probabilities, greedy_probabilities)
    return probabilities, greedy_tokens


def categorical_sample(
    probabilities: torch.Tensor, uniforms: torch.Tensor
) -> torch.Tensor:
    """First token whose inclusive CDF reaches the row's uniform."""
    cumulative = torch.cumsum(probabilities, dim=-1)
    return torch.sum(
        cumulative < uniforms.unsqueeze(-1), dim=-1, dtype=torch.int64
    ).clamp(max=probabilities.shape[-1] - 1)


def strict_categorical_sample(
    probabilities: torch.Tensor, uniforms: torch.Tensor
) -> torch.Tensor:
    """First supported token whose CDF exceeds the uniform, else the last one."""
    vocab_size = probabilities.shape[-1]
    supported = probabilities > 0.0
    token_ids = torch.arange(vocab_size, device=probabilities.device)
    crossing = torch.where(
        supported & (torch.cumsum(probabilities, dim=-1) > uniforms.unsqueeze(-1)),
        token_ids,
        torch.full_like(token_ids, vocab_size),
    ).amin(dim=-1)
    last_supported = torch.where(
        supported, token_ids, torch.zeros_like(token_ids)
    ).amax(dim=-1)
    return torch.where(crossing < vocab_size, crossing, last_supported)


def sample_tokens(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    uniforms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities, greedy_tokens = sampling_probabilities(
        logits, temperature, top_k, top_p
    )
    tokens = torch.where(
        temperature > 0.0, categorical_sample(probabilities, uniforms), greedy_tokens
    )
    return tokens, probabilities


def verify_speculative(
    target_logits: torch.Tensor,
    draft_probabilities: torch.Tensor,
    candidates: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    draft_argmax: torch.Tensor,
    accept_uniforms: torch.Tensor,
    correction_uniform: torch.Tensor,
) -> torch.Tensor:
    """Verifies ``R`` proposals against ``R + 1`` target rows.

    Returns ``[committed_count, correction_token, *candidates]`` so the host
    reads one cycle result with a single copy.
    """
    num_proposals = draft_probabilities.shape[0]
    target_probabilities, target_greedy = sampling_probabilities(
        target_logits, temperature, top_k, top_p
    )
    proposals = candidates[1:]
    proposal_index = proposals.unsqueeze(-1)
    p = target_probabilities[:num_proposals].gather(-1, proposal_index).squeeze(-1)
    q = torch.where(
        draft_argmax,
        torch.ones_like(p),
        draft_probabilities.gather(-1, proposal_index).squeeze(-1),
    )
    acceptance = torch.where(q > 0.0, torch.clamp(p / q, max=1.0), torch.ones_like(p))
    accepted = torch.where(
        temperature > 0.0,
        accept_uniforms < acceptance,
        target_greedy[:num_proposals] == proposals,
    )

    rows = torch.arange(num_proposals, device=candidates.device)
    # Row of the first rejection, or num_proposals (the bonus row) if none.
    decision_row = (
        torch.where(accepted, torch.full_like(rows, num_proposals), rows)
        .amin()
        .reshape(1)
    )
    rejected = decision_row < num_proposals
    proposal_row = decision_row.clamp(max=num_proposals - 1)

    row_target = target_probabilities.index_select(0, decision_row).squeeze(0)
    row_draft = draft_probabilities.index_select(0, proposal_row).squeeze(0)

    excluded = row_target.scatter(0, proposals.index_select(0, proposal_row), 0.0)
    excluded_sum = excluded.sum()
    excluded = excluded / torch.where(
        excluded_sum > 0.0, excluded_sum, torch.ones_like(excluded_sum)
    )
    residual = torch.clamp_min(row_target - row_draft, 0.0)
    residual_sum = residual.sum()
    residual = torch.where(
        residual_sum > 0.0,
        residual
        / torch.where(residual_sum > 0.0, residual_sum, torch.ones_like(residual_sum)),
        row_target,
    )

    rejection_correction = torch.where(
        draft_argmax,
        strict_categorical_sample(excluded.unsqueeze(0), correction_uniform),
        categorical_sample(residual.unsqueeze(0), correction_uniform),
    )
    stochastic_correction = torch.where(
        rejected,
        rejection_correction,
        categorical_sample(row_target.unsqueeze(0), correction_uniform),
    )
    correction = torch.where(
        temperature > 0.0,
        stochastic_correction,
        target_greedy.index_select(0, decision_row),
    )
    return torch.cat([decision_row + 1, correction, candidates])


class DFlashTokenSampler(nn.Module):
    """Samples ``[R, V]`` logits; returns ``[R]`` tokens and ``[R, V]`` probs."""

    def forward(
        self,
        logits: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        uniforms = torch.rand_like(logits[:, 0], dtype=torch.float32)
        return sample_tokens(logits, temperature, top_k, top_p, uniforms)


class DFlashSpeculativeVerifier(nn.Module):
    """Verifies ``R`` proposals; see :func:`verify_speculative`."""

    def forward(
        self,
        target_logits: torch.Tensor,
        draft_probabilities: torch.Tensor,
        candidates: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        draft_argmax: torch.Tensor,
    ) -> torch.Tensor:
        accept_uniforms = torch.rand_like(draft_probabilities[:, 0])
        correction_uniform = torch.rand_like(target_logits[:1, 0], dtype=torch.float32)
        return verify_speculative(
            target_logits,
            draft_probabilities,
            candidates,
            temperature,
            top_k,
            top_p,
            draft_argmax,
            accept_uniforms,
            correction_uniform,
        )
