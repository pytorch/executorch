# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.leaky_relu.default` module for the WebGPU op-test framework.

LeakyReLU is the activation in Real-ESRGAN's SRVGGNetCompact body.
"""

import torch


class LeakyReluModule(torch.nn.Module):
    def __init__(self, negative_slope: float) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.leaky_relu(x, self.negative_slope)
