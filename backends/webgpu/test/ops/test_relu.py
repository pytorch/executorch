# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.relu.default` module for the WebGPU op-test framework.

ReLU is on the SAM2/SAM3 mask-decoder MLP path.
"""

import torch


class ReluModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


def _det_input(shape) -> torch.Tensor:
    # ((i % 23) - 11) / 16: exact in fp32, spans negatives through positives
    # so the clamp is exercised, not just an identity pass-through.
    n = 1
    for s in shape:
        n *= s
    idx = torch.arange(n, dtype=torch.int64)
    return (((idx % 23) - 11).to(torch.float32) / 16.0).reshape(shape)
