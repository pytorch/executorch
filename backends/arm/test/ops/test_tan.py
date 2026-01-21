# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

input_t = Tuple[torch.Tensor]

aten_op = "torch.ops.aten.tan.default"
exir_op = "executorch_exir_dialects_edge__ops_aten__tan_default"

eps32 = torch.finfo(torch.float32).eps
tiny32 = torch.finfo(torch.float32).tiny

test_data_suite = {
    "zeros": torch.zeros(1, 10, 10, 10),
    "zeros_alt_shape": torch.zeros(1, 10, 3, 5),
    "ones": torch.ones(10, 15, 25),
    "rand": torch.rand(10, 10) - 0.5,
    "rand_alt_shape": torch.rand(1, 10, 3, 5) - 0.5,
    "randn_pos": torch.randn(10) + 10,
    "randn_neg": torch.randn(10) - 10,
    "ramp": torch.arange(-16, 16, 0.2),
    "pi_multiples": (torch.arange(-5, 6, dtype=torch.float32) * math.pi),
    "common_angles": torch.tensor(
        [
            -math.pi,
            -2 * math.pi / 3,
            -math.pi / 2 + 1e-3,
            -math.pi / 3,
            -math.pi / 4,
            -math.pi / 6,
            0.0,
            math.pi / 6,
            math.pi / 4,
            math.pi / 3,
            math.pi / 2 - 1e-3,
            2 * math.pi / 3,
            math.pi,
        ],
        dtype=torch.float32,
    ),
    "near_asymptote_pos": torch.tensor(
        [
            math.pi / 2 - 1e-7,
            math.pi / 2 - 1e-6,
            math.pi / 2 - 1e-4,
            math.pi / 2 + 1e-7,
            math.pi / 2 + 1e-6,
            math.pi / 2 + 1e-4,
        ],
        dtype=torch.float32,
    ),
    "high_rank": torch.randn(1, 3, 7, 4, 5),
    "very_small": torch.tensor(
        [-tiny32, -eps32, -1e-10, 0.0, 1e-10, eps32, tiny32], dtype=torch.float32
    ),
    "large_values": torch.linspace(-1e6, 1e6, steps=257, dtype=torch.float32),
    "undefined": torch.tensor([math.pi / 2, -math.pi / 2, 3 * math.pi / 2]),
}


class Tan(torch.nn.Module):

    def forward(self, x: torch.Tensor):
        return torch.tan(x)


@common.parametrize("test_data", test_data_suite)
def test_tan_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t](
        Tan(),
        (test_data,),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_tan_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t](
        Tan(),
        (test_data,),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_tan_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t](
        Tan(),
        (test_data,),
        aten_ops=aten_op,
        exir_ops=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_tan_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t](
        Tan(),
        (test_data,),
        aten_ops=aten_op,
        exir_ops=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_tan_vgf_no_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t](Tan(), (test_data,), [], [], quantize=False)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_tan_vgf_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t](
        Tan(),
        (test_data,),
        [],
        [],
        quantize=True,
    )
    pipeline.run()
