# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# pyre-unsafe
# LICENSE file in the root directory of this source tree.

"""Gemma 4 RMSNorm — self-contained re-implementation.

Numerically identical to ``transformers.models.gemma4.modeling_gemma4.Gemma4RMSNorm``
(same float32 upcast and ``pow(mean_squared, -0.5)`` normalization), but
without the transformers import so this module is exportable and dep-light.
"""

from functools import partial

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Gemma4 RMSNorm: ``y = (x / rms(x)) * weight``, computed in float32.

    Unlike Gemma 2/3 (``(1 + weight)``) Gemma 4 multiplies by ``weight`` directly.
    Pass ``with_scale=False`` for the v-norm and the (unused-here) router norm,
    which omit the learnable weight entirely.
    """

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # Match transformers' use of pow(mean_squared, -0.5) over rsqrt;
        # the comment there cites Torch/JAX compiler differences.
        mean_squared = x.pow(2).mean(-1, keepdim=True) + self.eps
        return x * torch.pow(mean_squared, -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self._norm(x.float())
        if self.with_scale:
            normed = normed * self.weight.float()
        return normed.type_as(x)


# V-norm in attention uses RMSNorm without learnable weight.
RMSNormNoWeight = partial(RMSNorm, with_scale=False)
