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

aten_op = "torch.ops.aten.cosh.default"
exir_op = "executorch_exir_dialects_edge__ops_aten__cosh_default"

input_t1 = Tuple[torch.Tensor]  # Input x

test_data_suite = {
    # (test_name, test_data)
    "zeros": lambda: torch.zeros(10, 10, 10),
    "zeros_4D": lambda: torch.zeros(1, 10, 32, 7),
    "zeros_alt_shape": lambda: torch.zeros(10, 3, 5),
    "ones": lambda: torch.ones(15, 10, 7),
    "ones_4D": lambda: torch.ones(1, 3, 32, 16),
    "rand": lambda: torch.rand(10, 10) - 0.5,
    "rand_alt_shape": lambda: torch.rand(10, 3, 5) - 0.5,
    "rand_4D": lambda: torch.rand(1, 6, 5, 7) - 0.5,
    "randn_pos": lambda: torch.randn(10) + 3,
    "randn_neg": lambda: torch.randn(10) - 3,
    "ramp": lambda: torch.arange(-16, 16, 0.2),
    "large": lambda: 100 * torch.ones(1, 1),
    "small": lambda: 0.000001 * torch.ones(1, 1),
    "small_rand": lambda: torch.rand(100) * 0.01,
    "biggest": lambda: torch.tensor([700.0, 710.0, 750.0]),
}


class Cosh(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.cosh(x)


@common.parametrize("test_data", test_data_suite)
def test_cosh_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        Cosh(),
        (test_data(),),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_cosh_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        Cosh(), (test_data(),), aten_op=aten_op, exir_op=exir_op
    )
    pipeline.run()


@common.XfailIfNoCorstone300
@common.parametrize("test_data", test_data_suite)
def test_cosh_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        Cosh(), (test_data(),), aten_ops=aten_op, exir_ops=exir_op
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
        Cosh(), (test_data(),), aten_ops=aten_op, exir_ops=exir_op
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_cosh_vgf_no_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Cosh(),
        (test_data(),),
        [],
        [],
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_cosh_vgf_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Cosh(),
        (test_data(),),
        [],
        [],
        quantize=True,
    )
    pipeline.run()
