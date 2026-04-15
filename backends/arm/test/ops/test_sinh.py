# Copyright 2025 Arm Limited and/or its affiliates.
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

aten_op = "torch.ops.aten.sinh.default"
exir_op = "executorch_exir_dialects_edge__ops_aten__sinh_default"


input_t1 = Tuple[torch.Tensor]  # Input x

test_data_suite = {
    # (test_name, test_data)
    "zeros": torch.zeros(10, 10, 10),
    "zeros_alt_shape": torch.zeros(10, 3, 5),
    "ones": torch.ones(10, 10, 10),
    "rand": torch.rand(10, 10) - 0.5,
    "rand_alt_shape": torch.rand(10, 3, 5) - 0.5,
    "randn_pos": torch.randn(10) + 10,
    "randn_neg": torch.randn(10) - 10,
    "ramp": torch.arange(-16, 16, 0.2),
    "large": 100 * torch.ones(1, 1),
    "small": 0.000001 * torch.ones(1, 1),
}


class Sinh(torch.nn.Module):

    def forward(self, x: torch.Tensor):
        return torch.sinh(x)


@common.parametrize("test_data", test_data_suite)
def test_sinh_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        Sinh(),
        (test_data,),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_sinh_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        Sinh(), (test_data,), aten_op=aten_op, exir_op=exir_op
    )
    pipeline.run()


@common.XfailIfNoCorstone300
@common.parametrize("test_data", test_data_suite)
def test_sinh_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        Sinh(), (test_data,), aten_ops=aten_op, exir_ops=exir_op
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("test_data", test_data_suite)
def test_sinh_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        Sinh(), (test_data,), aten_ops=aten_op, exir_ops=exir_op
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_sinh_vgf_no_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Sinh(),
        (test_data,),
        aten_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_sinh_vgf_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Sinh(),
        (test_data,),
        aten_op,
        quantize=True,
    )
    pipeline.run()
