# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.pow.Tensor_Tensor` module for the WebGPU op-test framework.

`PowModule` is imported by `cases.py`. The suite covers same-shape and broadcast
`pow(a, b)` with POSITIVE bases so `pow(neg, frac)` never produces NaN.
"""

import torch


class PowModule(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.pow(a, b)
