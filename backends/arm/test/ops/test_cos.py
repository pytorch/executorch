# Copyright 2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest

import torch
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op = "torch.ops.aten.cos.default"
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


class Cos(torch.nn.Module):

    def forward(self, x: torch.Tensor):
        return torch.cos(x)


@common.parametrize("test_data", test_data_suite)
@pytest.mark.tosa_ref_model
def test_cos_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        Cos(),
        (test_data,),
        aten_op,
        exir_op=[],
        run_on_tosa_ref_model=conftest.is_option_enabled("tosa_ref_model"),
    )
    if conftest.get_option("tosa_version") == "1.0":
        pipeline.run()


@common.parametrize("test_data", test_data_suite)
@pytest.mark.tosa_ref_model
def test_cos_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        Cos(),
        (test_data,),
        aten_op,
        exir_op=[],
        run_on_tosa_ref_model=conftest.is_option_enabled("tosa_ref_model"),
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_cos_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        Cos(),
        (test_data,),
        aten_op,
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_cos_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        Cos(),
        (test_data,),
        aten_op,
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_cos_vgf_FP(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Cos(),
        (test_data,),
        aten_op,
        exir_op=[],
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_cos_vgf_INT(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Cos(),
        (test_data,),
        aten_op,
        exir_op=[],
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
