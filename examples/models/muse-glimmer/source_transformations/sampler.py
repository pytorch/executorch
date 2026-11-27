# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU-side sampler: a Gumbel-max sample, or the argmax at temperature 0.

A single-output sampler that lets one exported program be re-driven with
different temperatures without re-export.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F


def sample(
    logits: torch.Tensor,
    temperature: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Draw one token per row: a Gumbel-max sample, or the argmax at temperature 0.

    Args:
        logits: ``[..., V]`` logits (already soft-capped).
        temperature: 0-D or shape ``[1]`` float tensor, one temperature for
            every row. Above 0, the token is a Gumbel-max sample of
            ``logits / temperature``, with temperatures below 1e-6 taken as
            1e-6; at 0 or below, the argmax of ``logits``. When ``None``,
            returns ``logits`` unchanged.

    Returns:
        ``[..., 1]`` float32 token IDs, or unmodified logits when
        ``temperature`` is ``None``.
    """
    if temperature is None:
        return logits
    return sample_token(logits, temperature).float()


def sample_token(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """``sample`` as ``[..., 1]`` int64 token IDs.

    Temperature 0 takes the argmax of ``logits`` itself rather than of logits
    divided by a tiny temperature, where the noise would still flip near-ties
    from run to run. The graph is the same at every temperature, so the noise
    is drawn either way. The clamp puts a floor of 1e-6 under positive
    temperatures.
    """
    scaled = logits / temperature.clamp(min=1e-6)
    noise = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
    return argmax_last_dim(torch.where(temperature > 0, scaled + gumbel, logits))


def argmax_last_dim(values: torch.Tensor) -> torch.Tensor:
    """``values.argmax(dim=-1, keepdim=True)`` as two reductions.

    Ties and NaN resolve as in ``torch.argmax``: the first index of the maximum
    wins within each tile, and the first such tile wins across tiles.
    """
    *batch, vocab = values.shape
    tile = _argmax_tile(vocab)
    tiles = -(-vocab // tile)
    rows = values.reshape(-1, vocab)
    padded = F.pad(rows, (0, tiles * tile - vocab), value=float("-inf"))
    tile_max, tile_arg = padded.view(-1, tiles, tile).max(dim=-1)
    best_tile = tile_max.argmax(dim=-1, keepdim=True)
    # A masked sum, not `gather`: Inductor guards an indirect load with a
    # device-side assert, whose first launch stalls the first prefill.
    tile_ids = torch.arange(tiles, device=values.device)
    offset = torch.where(tile_ids == best_tile, tile_arg, 0).sum(dim=-1, keepdim=True)
    return (best_tile * tile + offset).view(*batch, 1)


def _argmax_tile(vocab: int) -> int:
    """The power of two at or above sqrt(vocab).

    A single [B, V] argmax runs on one thread block per row. About sqrt(V)
    tiles of about sqrt(V) entries give the first reduction enough thread blocks
    and leave the second one short, for any vocabulary; the split depends on V
    alone, and Inductor picks each reduction's launch configuration for the GPU.
    """
    return 1 << math.isqrt(max(vocab - 1, 0)).bit_length()
