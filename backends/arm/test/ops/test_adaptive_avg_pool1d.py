# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Tests AdaptiveAvgPool1d through its standard 1D -> 2D decomposition path."""

from typing import Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    VgfPipeline,
)

input_t = Tuple[torch.Tensor]
aten_op = "torch.ops.aten.adaptive_avg_pool1d.default"


class AdaptiveAvgPool1d(torch.nn.AdaptiveAvgPool1d):
    pass


test_modules = {
    "output_1_from_8": lambda: (
        AdaptiveAvgPool1d(1),
        (torch.rand(1, 4, 8),),
    ),
    "output_4_from_16": lambda: (
        AdaptiveAvgPool1d(4),
        (torch.rand(1, 4, 16),),
    ),
    "non_divisible_5_from_17": lambda: (
        AdaptiveAvgPool1d(5),
        (torch.rand(1, 4, 17),),
    ),
}


@common.parametrize("test_module", test_modules)
def test_adaptive_avg_pool1d_tosa_FP(test_module):
    model, input_tensor = test_module()

    pipeline = TosaPipelineFP[input_t](
        model,
        input_tensor,
        aten_op=[aten_op],
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_module", test_modules)
@common.SkipIfNoModelConverter
def test_adaptive_avg_pool1d_vgf_no_quant(test_module):
    model, input_tensor = test_module()

    pipeline = VgfPipeline[input_t](
        model,
        input_tensor,
        aten_op=[aten_op],
        exir_op=[],
        quantize=False,
    )
    pipeline.run()
