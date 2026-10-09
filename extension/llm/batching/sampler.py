# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exportable per-row token sampler for batched decoding.

A batched step yields one logits row per sequence that samples, each with its
own policy. ``BatchSampler`` draws all of them in one call so the logits never
leave the device: an exporter adds it as a method next to the forward methods
and keeps the forward's logits output and the sampler's inputs device-resident
(``PropagateDeviceConfig``), and the runtime copies back only ``[R]`` tokens.

It mirrors the host sampler in ``extension/llm/sampler/sampler.cpp`` row by
row, so a runtime may use either:

- temperature 0 is exact argmax of the raw logits;
- otherwise softmax(logits / temperature), then top-k when 0 < k < vocab,
  else top-p when 0 < p < 1, else the full distribution;
- top-k and top-p sample in descending-probability order, keeping the tokens
  up to and including the one where the cumulative probability first exceeds
  p; the full distribution samples in token order;
- the uniform ``coin`` is an input, so the runtime draws it from the
  sequence's own seeded generator and a seed reproduces the generation.

``params`` packs each row's policy as float32 ``[temperature, top_p, top_k,
coin]`` so a step uploads one small tensor. Ties between equal probabilities
break toward the lower token id (stable sort), which the host's
``std::partial_sort`` does not specify.
"""

from __future__ import annotations

import torch
from torch import nn

PARAM_TEMPERATURE = 0
PARAM_TOP_P = 1
PARAM_TOP_K = 2
PARAM_COIN = 3
NUM_PARAMS = 4


def sample_rows(logits: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Draws one token per row of ``logits`` ``[R, V]``; returns int64 ``[R]``."""
    vocab_size = logits.shape[-1]
    logits = logits.float()
    temperature = params[:, PARAM_TEMPERATURE : PARAM_TEMPERATURE + 1]
    top_p = params[:, PARAM_TOP_P : PARAM_TOP_P + 1]
    top_k = params[:, PARAM_TOP_K : PARAM_TOP_K + 1]
    coin = params[:, PARAM_COIN : PARAM_COIN + 1]
    greedy = torch.argmax(logits, dim=-1)

    stochastic = temperature > 0.0
    safe_temperature = torch.where(
        stochastic, temperature, torch.ones_like(temperature)
    )
    probabilities = torch.softmax(logits / safe_temperature, dim=-1)

    sorted_probabilities, sorted_indices = torch.sort(
        probabilities, dim=-1, descending=True, stable=True
    )
    ranks = torch.arange(vocab_size, device=logits.device).unsqueeze(0)
    cumulative = torch.cumsum(sorted_probabilities, dim=-1)

    use_top_k = (top_k > 0.0) & (top_k < vocab_size)
    use_top_p = ~use_top_k & (top_p > 0.0) & (top_p < 1.0)
    # The kept prefix of the sorted distribution: k tokens for top-k; for
    # top-p, every token whose preceding mass has not yet exceeded p.
    kept = torch.where(
        use_top_k,
        ranks < top_k,
        torch.where(use_top_p, (cumulative - sorted_probabilities) <= top_p, True),
    )
    kept_probabilities = torch.where(
        kept, sorted_probabilities, torch.zeros_like(sorted_probabilities)
    )
    kept_cumulative = torch.cumsum(kept_probabilities, dim=-1)
    kept_mass = kept_cumulative[:, -1:]
    # First kept position whose cumulative mass passes coin * mass, else the
    # last kept one (rounding).
    passes = kept & (coin * kept_mass < kept_cumulative)
    last_kept = torch.sum(kept, dim=-1, keepdim=True) - 1
    first_pass = torch.where(passes, ranks, torch.full_like(ranks, vocab_size)).amin(
        dim=-1, keepdim=True
    )
    sorted_choice = torch.where(first_pass < vocab_size, first_pass, last_kept)
    truncated = sorted_indices.gather(-1, sorted_choice).squeeze(-1)

    # The full distribution samples in token order, as the host does.
    full_cumulative = torch.cumsum(probabilities, dim=-1)
    full = torch.sum(full_cumulative <= coin, dim=-1).clamp(max=vocab_size - 1)

    sampled = torch.where((use_top_k | use_top_p).squeeze(-1), truncated, full)
    return torch.where(stochastic.squeeze(-1), sampled, greedy)


class BatchSampler(nn.Module):
    """``(logits [R, V], params [R, 4] float32) -> tokens [R] int64``."""

    def forward(self, logits: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return sample_rows(logits, params)


class BatchArgmax(nn.Module):
    """``logits [R, V] -> tokens [R] int64``: the greedy-only sampler.

    A step whose rows are all greedy needs no sort over the vocabulary; a
    runtime may call this instead of ``BatchSampler`` for such steps.
    """

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits.float(), dim=-1)
