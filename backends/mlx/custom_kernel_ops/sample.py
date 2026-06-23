#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional

import torch
from torch import Tensor


@torch.library.custom_op("mlx::sample", mutates_args=())
def sample(
    logits: Tensor, temperature: Tensor, seed: Optional[Tensor] = None
) -> Tensor:
    """
    Gumbel-max sampling from softmax(logits / temperature).
    logits:      [B, vocab]
    temperature: scalar float tensor    (runtime input)
    seed:        scalar int tensor or None
                 - tensor -> deterministic, keyed RNG (random::key(seed))
                 - None   -> MLX global KeySequence (non-deterministic)
    -> token_id: [B] int64
    Reference (CPU) implementation for export + numerical parity.
    """
    if seed is None:
        u = torch.rand(logits.shape)  # global RNG
    else:
        gen = torch.Generator().manual_seed(int(seed.item()))
        u = torch.rand(logits.shape, generator=gen)
    gumbel = -torch.log(-torch.log(u))
    return torch.argmax(logits / temperature + gumbel, dim=-1)


@torch.library.register_fake("mlx::sample")
def sample_fake(logits, temperature, seed=None):
    return logits.new_empty(logits.shape[:-1], dtype=torch.long)
