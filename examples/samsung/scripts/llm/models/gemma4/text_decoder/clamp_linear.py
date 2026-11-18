# Copyright (c) 2026 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClampLinear(nn.Module):
    """Replaces a `QuantizedLinear` with clamp + linear + clamp."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        in_scale: torch.Tensor,
        out_scale: torch.Tensor,
        a_bits: int = 8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.a_bits = a_bits
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if bias is not None else None
        self.in_scale = in_scale
        self.out_scale = out_scale

        max_v = 2 ** (self.a_bits - 1) - 1
        min_v = -max_v - 1

        self.ifm_min, self.ifm_max = (min_v * in_scale).item(), (
            max_v * in_scale
        ).item()
        self.ofm_min, self.ofm_max = (min_v * out_scale).item(), (
            max_v * out_scale
        ).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, self.ifm_min, self.ifm_max)
        out = F.linear(x, self.weight, self.bias)
        out = torch.clamp(out, self.ofm_min, self.ofm_max)
        return out
