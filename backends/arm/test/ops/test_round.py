# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import pytest
import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

input_t1 = Tuple[torch.Tensor]  # Input x

aten_op = "torch.ops.aten.round.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_round_default"

test_data_suite = {
    # (test_name, test_data)
    "zeros": lambda: torch.zeros(1, 10, 10, 10),
    "ones": lambda: torch.ones(10, 10, 10),
    "rand": lambda: torch.rand(10, 10) - 0.5,
    "randn_pos": lambda: torch.randn(10) + 10,
    "randn_neg": lambda: torch.randn(10) - 10,
    "ramp": lambda: torch.arange(-16, 16, 0.2),
}


class Round(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x.round()


@common.parametrize("test_data", test_data_suite)
def test_round_tosa_FP(test_data: torch.Tensor):
    pipeline = TosaPipelineFP[input_t1](
        Round(),
        (test_data(),),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_round_tosa_INT(test_data: torch.Tensor):
    pipeline = TosaPipelineINT[input_t1](
        Round(),
        (test_data(),),
        [],
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
@pytest.mark.xfail(reason="where.self not supported on U55")
def test_round_u55_INT(test_data: torch.Tensor):
    pipeline = EthosU55PipelineINT[input_t1](
        Round(),
        (test_data(),),
        [],
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_round_u85_INT(test_data: torch.Tensor):
    pipeline = EthosU85PipelineINT[input_t1](
        Round(),
        (test_data(),),
        [],
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_round_vgf_no_quant(test_data: torch.Tensor):
    pipeline = VgfPipeline[input_t1](
        Round(),
        (test_data(),),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_round_vgf_quant(test_data: torch.Tensor):
    pipeline = VgfPipeline[input_t1](
        Round(),
        (test_data(),),
        [],
        exir_op,
        quantize=True,
    )
    pipeline.run()
