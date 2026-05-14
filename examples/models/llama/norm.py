# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, add_unit_offset: bool = False):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
            add_unit_offset (bool, optional): Whether to scale normalized output by
                `(1 + weight)` instead of `weight`. Default is False.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.add_unit_offset:
            return output * (1.0 + self.weight.float()).type_as(x)
        return output * self.weight.type_as(x)


class ScalelessRMSNorm(torch.nn.RMSNorm):
    """RMSNorm with weight hardcoded to ones and not trainable.

    Equivalent to a scaleless RMSNorm (no learnable scaling) but implemented as a
    torch.nn.RMSNorm so the op composes/decomposes cleanly for backends like QNN
    instead of being expressed as a hand-rolled decomposition.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(dim, eps)
        self.dim = dim
        with torch.no_grad():
            self.weight.fill_(1.0)
        self.weight.requires_grad = False


class RMSNormWithInputScale(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        scaled = self.weight * x
        return F.rms_norm(scaled, (self.dim,), None, self.eps)


class RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)
