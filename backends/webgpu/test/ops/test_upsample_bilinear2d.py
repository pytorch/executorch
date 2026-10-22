# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.upsample_bilinear2d.vec` module for the WebGPU op-test framework.

Bilinear upsample is on the Depth-Anything / DPT reassemble+fusion head
(`F.interpolate(..., mode="bilinear", align_corners=...)`), which resizes the
ViT patch grid back to image resolution.
"""

import torch


class UpsampleBilinear2dModule(torch.nn.Module):
    def __init__(self, scale_factor: float, align_corners: bool) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=self.align_corners,
        )
