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

input_t = Tuple[torch.Tensor]  # Input x
aten_op = "torch.ops.aten.asin.default"

test_data_suite = {
    "zeros": lambda: torch.zeros(1, 5, 3, 2),  # valid: asin(0) = 0
    "ones": lambda: torch.ones(10, 5, 15),  # edge case: asin(1) = pi/2
    "neg_ones": lambda: -torch.ones(10, 5, 15),  # edge case: asin(-1) = -pi/2
    "rand": lambda: (torch.rand(10, 10, 5) * 2) - 1,  # uniform random in [-1, 1]
    "ramp": lambda: torch.linspace(-1.0, 1.0, steps=160),  # full domain coverage
    "near_bounds": lambda: torch.tensor(
        [-0.999, -0.9, -0.5, 0.0, 0.5, 0.9, 0.999]
    ),  # precision edge values
    "pos_rand": lambda: torch.rand(7, 10, 2),  # positive random values in [0, 1]
}


class Asin(torch.nn.Module):
    def forward(self, x):
        return torch.asin(x)


@common.parametrize("test_data", test_data_suite)
def test_asin_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t](
        Asin(),
        (test_data(),),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_asin_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t](
        Asin(),
        (test_data(),),
        aten_op=[],
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_asin_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t](
        Asin(),
        (test_data(),),
        aten_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_asin_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t](
        Asin(),
        (test_data(),),
        aten_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_asin_vgf_no_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t](
        Asin(),
        (test_data(),),
        aten_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_asin_vgf_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t](
        Asin(),
        (test_data(),),
        aten_op,
        quantize=True,
    )
    pipeline.run()
