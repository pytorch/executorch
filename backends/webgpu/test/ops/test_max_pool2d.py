# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.max_pool2d_with_indices.default` module for the WebGPU op-test
framework.

`F.max_pool2d` decomposes to `aten.max_pool2d_with_indices.default`, a
multi-output op whose out is a ValueList `[values, indices]`; the handler
writes the VALUES only and never the int64 indices
(`runtime/ops/max_pool2d/MaxPool2d.cpp`). Max-pool is on the SAM2 Hiera
q_pool path.
"""

import torch


class MaxPool2dModule(torch.nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )


def _det_input(shape) -> torch.Tensor:
    # ((i % 23) - 11) / 16: exact in fp32, spans negatives through positives.
    n = 1
    for s in shape:
        n *= s
    idx = torch.arange(n, dtype=torch.int64)
    return (((idx % 23) - 11).to(torch.float32) / 16.0).reshape(shape)
