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

aten_op = "torch.ops.aten.sign.default"
exir_op = "executorch_exir_dialects_edge__ops_aten__sign_default"

input_t1 = Tuple[torch.Tensor]

test_data_suite = {
    "zeros": torch.zeros(3, 5),
    "ones": torch.ones(4, 4),
    "neg_ones": -torch.ones(4, 4),
    "mixed_signs": torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]]),
    "positive_ramp": torch.arange(0.1, 1.1, 0.2),
    "negative_ramp": torch.arange(-1.0, -0.1, 0.2),
    "small_values": torch.tensor([-1e-7, 0.0, 1e-7]),
    "rand": torch.rand(10, 10) - 0.5,
    "rand_alt_shape": torch.rand(10, 3, 5) - 0.5,
    "high_magnitude": torch.tensor([-1e6, -10.0, 0.0, 10.0, 1e6]),
}


class Sign(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.sign(x)


@common.parametrize("test_data", test_data_suite)
def test_sign_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        Sign(),
        (test_data,),
        aten_op=aten_op,
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_sign_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        Sign(),
        (test_data,),
        aten_op=[],
        exir_op=exir_op,
    )
    pipeline.run()


@common.XfailIfNoCorstone300
@common.parametrize("test_data", test_data_suite)
@pytest.mark.xfail(reason="where.self not supported on U55")
def test_sign_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        Sign(),
        (test_data,),
        aten_ops=[],
        exir_ops=exir_op,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("test_data", test_data_suite)
def test_sign_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        Sign(),
        (test_data,),
        aten_ops=[],
        exir_ops=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_sign_vgf_no_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Sign(),
        (test_data,),
        aten_op=aten_op,
        exir_op=exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_sign_vgf_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Sign(),
        (test_data,),
        aten_op=[],
        exir_op=exir_op,
        quantize=True,
    )
    pipeline.run()
