# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.minimum.default` module for the WebGPU op-test framework.

`MinimumModule` is imported by `cases.py`. minimum is a same-shape elementwise
binary op mirroring the landed `add`/`mul` pattern (flat 2D-dispatch kernel).
"""

import torch


class MinimumModule(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.minimum(a, b)
