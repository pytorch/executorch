# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU-side Gumbel-max sampler.

Mirrors ``examples/models/qwen3_5_moe/sampler.py``: a single-output sampler
that lets one exported program be re-driven with different temperatures
without re-export.
"""

import torch


def sample(
    logits: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Draw a single token per batch row using the Gumbel-max trick.

    Args:
        logits: ``[B, V]`` float32 logits (already soft-capped if applicable).
        temperature: 0-D or 1-D float tensor; clamped to >= 1e-6 so a 0
            temperature still works ("near-greedy").

    Returns:
        ``[B, 1]`` int64 token IDs (``argmax(logits/T + gumbel_noise)``).
        Emitting int64 (rather than casting to float) lets the runner alias the
        on-device output token directly as the next decode step's int64 token
        input — no D2H/H2D round-trip and no dtype cast.
    """
    logits = logits / temperature.clamp(min=1e-6)
    noise = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
    return (logits + gumbel).argmax(dim=-1, keepdim=True).to(torch.int64)
