# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU-side Gumbel-max sampler.

A single-output sampler that lets one exported program be re-driven with
different temperatures without re-export.
"""

from typing import Optional

import torch


def sample(
    logits: torch.Tensor,
    temperature: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Draw a single token per batch row using the Gumbel-max trick.

    Args:
        logits: ``[B, V]`` float32 logits (already soft-capped).
        temperature: 0-D or 1-D float tensor; clamped to >= 1e-6.
            When ``None``, returns ``logits`` unchanged.

    Returns:
        ``[B, 1]`` float32 token IDs, or unmodified logits when
        ``temperature`` is ``None``.
    """
    if temperature is None:
        return logits

    logits = logits / temperature.clamp(min=1e-6)
    noise = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
    return (logits + gumbel).argmax(dim=-1, keepdim=True).float()
