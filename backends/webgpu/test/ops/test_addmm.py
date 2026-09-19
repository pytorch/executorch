# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.addmm.default` module for the WebGPU op-test framework.

HF Linear lowers to addmm (not aten.linear), so this is the dominant GEMM in the
Florence-2 BART + DaViT graphs: out = beta*self + alpha*(mat1 @ mat2).
"""

import torch


class AddmmModule(torch.nn.Module):
    def __init__(self, n: int, beta: float = 1.0, alpha: float = 1.0):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.linspace(-0.5, 0.5, n))
        self.beta, self.alpha = beta, alpha

    def forward(self, mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
        return torch.addmm(self.bias, mat1, mat2, beta=self.beta, alpha=self.alpha)


def _randn(shape) -> torch.Tensor:
    g = torch.Generator().manual_seed(sum(int(x) for x in shape))
    return torch.randn(*shape, generator=g) * 0.1
