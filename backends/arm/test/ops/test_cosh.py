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

aten_op = "torch.ops.aten.cosh.default"
exir_op = "executorch_exir_dialects_edge__ops_aten__cosh_default"

input_t1 = Tuple[torch.Tensor]  # Input x

test_data_suite = {
    # (test_name, test_data)
    "zeros": torch.zeros(10, 10, 10),
    "zeros_4D": torch.zeros(1, 10, 32, 7),
    "zeros_alt_shape": torch.zeros(10, 3, 5),
    "ones": torch.ones(15, 10, 7),
    "ones_4D": torch.ones(1, 3, 32, 16),
    "rand": torch.rand(10, 10) - 0.5,
    "rand_alt_shape": torch.rand(10, 3, 5) - 0.5,
    "rand_4D": torch.rand(1, 6, 5, 7) - 0.5,
    "randn_pos": torch.randn(10) + 10,
    "randn_neg": torch.randn(10) - 10,
    "ramp": torch.arange(-16, 16, 0.2),
    "large": 100 * torch.ones(1, 1),
    "small": 0.000001 * torch.ones(1, 1),
    "small_rand": torch.rand(100) * 0.01,
    "biggest": torch.tensor([700.0, 710.0, 750.0]),
}


class Cosh(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.cosh(x)


@common.parametrize("test_data", test_data_suite)
def test_cosh_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        Cosh(),
        (test_data,),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_cosh_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        Cosh(), (test_data,), aten_op=aten_op, exir_op=exir_op
    )
    pipeline.run()


@common.XfailIfNoCorstone300
@common.parametrize("test_data", test_data_suite)
def test_cosh_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        Cosh(), (test_data,), aten_ops=aten_op, exir_ops=exir_op
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize(
    "test_data",
    test_data_suite,
    strict=False,
)
def test_cosh_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        Cosh(), (test_data,), aten_ops=aten_op, exir_ops=exir_op
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_cosh_vgf_FP(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Cosh(),
        (test_data,),
        [],
        [],
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_cosh_vgf_INT(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Cosh(),
        (test_data,),
        [],
        [],
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
