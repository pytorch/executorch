# Copyright 2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op = "torch.ops.aten.sin.default"
input_t1 = Tuple[torch.Tensor]  # Input x

test_data_suite = {
    # (test_name, test_data)
    "zeros": torch.zeros(10, 10, 10, 10),
    "ones": torch.ones(10, 10, 10),
    "rand": torch.rand(10, 10) - 0.5,
    "randn_pos": torch.randn(10) + 10,
    "randn_neg": torch.randn(10) - 10,
    "ramp": torch.arange(-16, 16, 0.2),
}


class Sin(torch.nn.Module):

    def forward(self, x: torch.Tensor):
        return torch.sin(x)


@common.parametrize("test_data", test_data_suite)
def test_sin_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        Sin(),
        (test_data,),
        aten_op,
        exir_op=[],
    )
    if conftest.get_option("tosa_version") == "1.0":
        pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_sin_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        Sin(),
        (test_data,),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_sin_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        Sin(),
        (test_data,),
        aten_op,
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_sin_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        Sin(),
        (test_data,),
        aten_op,
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_sin_vgf_FP(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Sin(), (test_data,), aten_op, tosa_version="TOSA-1.0+FP"
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_sin_vgf_INT(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Sin(),
        (test_data,),
        aten_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
