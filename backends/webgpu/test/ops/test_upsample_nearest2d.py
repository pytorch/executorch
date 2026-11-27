# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.upsample_nearest2d.vec` module for the WebGPU op-test framework.

Upsample is on the SAM2/SAM3 pixel-decoder / FPN path
(`F.interpolate(..., mode="nearest")`).
"""

import torch


class UpsampleNearest2dModule(torch.nn.Module):
    def __init__(self, scale_factor: float) -> None:
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode="nearest"
        )


def _det_input(shape) -> torch.Tensor:
    # ((i % 23) - 11) / 16: exact in fp32, spans negatives through positives.
    n = 1
    for s in shape:
        n *= s
    idx = torch.arange(n, dtype=torch.int64)
    return (((idx % 23) - 11).to(torch.float32) / 16.0).reshape(shape)
