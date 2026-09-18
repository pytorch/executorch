# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Block-local convolution and candidate scoring for DFlash2."""

import torch
from torch import nn
from torch.nn import functional as F


def candidate_topk(logits: torch.Tensor, top_k: int):
    """Exact global top-k using chunks supported by the CUDA top-k kernel."""
    chunk_size = 4096
    vocab_size = logits.shape[-1]
    if vocab_size <= chunk_size:
        return torch.topk(logits, top_k, dim=-1)
    chunks = (vocab_size + chunk_size - 1) // chunk_size
    padded = F.pad(logits, (0, chunks * chunk_size - vocab_size), value=-float("inf"))
    values, ids = torch.topk(padded.unflatten(-1, (chunks, chunk_size)), top_k, dim=-1)
    offsets = torch.arange(chunks, device=logits.device)[:, None] * chunk_size
    ids = (ids + offsets).flatten(-2)
    values, selected = torch.topk(values.flatten(-2), top_k, dim=-1)
    return values, ids.gather(-1, selected)


class DFlashGroupedConv(nn.Module):
    def __init__(self, hidden_size: int, taps: int, group_size: int):
        super().__init__()
        if taps < 1 or group_size < 1 or hidden_size % group_size:
            raise ValueError(
                "Convolution taps must be positive and groups must divide hidden_size"
            )
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        base = torch.zeros(2, taps, hidden_size)
        base[:, 0] = 1
        self.base_kernel = nn.Parameter(base)
        self.kernel_projection = nn.Linear(
            hidden_size, 2 * taps * self.num_groups, bias=False
        )

    def _convolve(
        self, x: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        blocks = x.float().unflatten(-1, (self.num_groups, self.group_size))
        coefficients = self.base_kernel[side].float().view(
            1, 1, self.taps, self.num_groups, self.group_size
        ) + delta.float().unsqueeze(-1)
        output = coefficients[:, :, 0] * blocks
        for tap in range(1, self.taps):
            shifted = F.pad(blocks, (0, 0, 0, 0, tap, 0))[:, : x.shape[1]]
            output = output + coefficients[:, :, tap] * shifted
        return output.flatten(-2).to(x.dtype)

    def prepare(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.kernel_projection(x).unflatten(
            -1, (2, self.taps, self.num_groups)
        )
        return self._convolve(x, coefficients[:, :, 0], 0), coefficients[:, :, 1]

    def finish(self, x: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        return self._convolve(x, coefficients, 1)


class CandidateSelector(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, rank: int, top_k: int):
        super().__init__()
        if rank < 1 or not 1 <= top_k <= vocab_size:
            raise ValueError(
                "Selector rank must be positive and top_k must fit the vocabulary"
            )
        self.top_k = top_k
        self.predecessor_codebook = nn.Embedding(vocab_size, rank)
        self.successor_codebook = nn.Embedding(vocab_size, rank)
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        predecessor_ids = torch.cat(
            (
                anchor_token_ids[:, None, None].expand(-1, 1, self.top_k),
                candidate_ids[:, :-1],
            ),
            dim=1,
        )
        predecessors = self.predecessor_codebook(predecessor_ids)
        successors = self.successor_codebook(candidate_ids)
        hidden = self.hidden_projection(hidden_states)
        pair_scores = torch.einsum(
            "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
        )
        return unary_logits.float()[:, :, None] + pair_scores.float()
