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
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
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
def test_cos_tosa_MI(test_data: Tuple):
    pipeline = TosaPipelineMI[input_t1](
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
def test_cos_tosa_BI(test_data: Tuple):
    pipeline = TosaPipelineBI[input_t1](
        Cos(),
        (test_data,),
        aten_op,
        exir_op=[],
        run_on_tosa_ref_model=conftest.is_option_enabled("tosa_ref_model"),
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_cos_tosa_u55_BI(test_data: Tuple):
    pipeline = EthosU55PipelineBI[input_t1](
        Cos(),
        (test_data,),
        aten_op,
        exir_ops=[],
        run_on_fvp=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_cos_tosa_u85_BI(test_data: Tuple):
    pipeline = EthosU85PipelineBI[input_t1](
        Cos(),
        (test_data,),
        aten_op,
        exir_ops=[],
        run_on_fvp=False,
    )
    pipeline.run()
