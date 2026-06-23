#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn


class SamplingHead(nn.Module):
    """
    Wraps a model that returns logits and samples a token id on-device.

        forward(*model_args, temperature, seed=None, **model_kwargs) -> token_id

      temperature: scalar float tensor, e.g. torch.tensor(0.8). Must be > 0;
                   logits are divided by it, so 0.0 yields inf/nan. For greedy,
                   pass a small epsilon (e.g. 1e-4), not 0.
      seed:        scalar int tensor (seeded) or None (unseeded export)
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, temperature, seed=None, **kwargs):
        logits = self.model(*args, **kwargs)  # [B, S, vocab]
        last = logits[:, -1, :]  # [B, vocab]
        return torch.ops.mlx.sample(last, temperature, seed)
