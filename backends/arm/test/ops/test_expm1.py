# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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

aten_op = "torch.ops.aten.expm1.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_expm1_default"

input_t1 = Tuple[torch.Tensor]

test_data_suite = {
    "zeroes": lambda: torch.zeros(1, 10, 10, 10),
    "ones": lambda: torch.ones(10, 2, 3),
    "rand": lambda: torch.rand(10, 10) - 0.5,
    "near_zero": lambda: torch.randn(100) * 0.01,
    "taylor_small": lambda: torch.empty(5).uniform_(
        -0.35, 0.35
    ),  # test cases for taylor series expansion
    "randn_large_pos": lambda: torch.randn(10) + 10,
    "randn_large_neg": lambda: torch.randn(10) - 10,
    "ramp": lambda: torch.arange(-16, 16, 0.2),
}


class Expm1(torch.nn.Module):

    def forward(self, x: torch.Tensor):
        return torch.expm1(x)


@common.parametrize("test_data", test_data_suite)
def test_expm1_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        Expm1(),
        (test_data(),),
        aten_op=aten_op,
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_expm1_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        Expm1(),
        (test_data(),),
        aten_op=aten_op,
        exir_op=exir_op,
    )
    pipeline.run()


@common.XfailIfNoCorstone300
@common.parametrize("test_data", test_data_suite)
def test_expm1_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        Expm1(),
        (test_data(),),
        aten_ops=aten_op,
        exir_ops=exir_op,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("test_data", test_data_suite)
def test_expm1_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        Expm1(),
        (test_data(),),
        aten_ops=aten_op,
        exir_ops=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_expm1_vgf_no_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Expm1(),
        (test_data(),),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_expm1_vgf_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Expm1(),
        (test_data(),),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()
