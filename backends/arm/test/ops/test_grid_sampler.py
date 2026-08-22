# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn.functional as F
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import VgfPipeline

input_t = Tuple[torch.Tensor, torch.Tensor]
aten_op = "torch.ops.aten.grid_sampler.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_grid_sampler_2d_default"

test_data_suite = {
    "2d_bilinear_zeros": lambda: (
        torch.randn(1, 3, 8, 8),
        torch.randn(1, 4, 4, 2),
    ),
}

xfails = {
    "2d_bilinear_zeros": (
        "CI model_converter does not yet include Vulkan custom-shader "
        "tosa.custom legalization",
        RuntimeError,
    ),
}


class GridSampler2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.interpolation_mode_ = 0
        self.padding_mode_ = 0
        self.align_corners_ = False

    def forward(self, x, grid):
        return F.grid_sample(
            x,
            grid,
            mode="bilinear" if self.interpolation_mode_ == 0 else "nearest",
            padding_mode="zeros" if self.padding_mode_ == 0 else "border",
            align_corners=self.align_corners_,
        )


@common.parametrize("test_data", test_data_suite, xfails=xfails, strict=False)
@common.SkipIfNoModelConverter
def test_grid_sampler_vgf_no_quant(test_data):
    test_data = test_data()
    pipeline = VgfPipeline[input_t](
        GridSampler2d(),
        test_data,
        aten_op,
        exir_op,
        quantize=False,
        run_on_vulkan_runtime=False,
    )
    pipeline.run()
