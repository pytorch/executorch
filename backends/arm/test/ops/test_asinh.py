# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)

input_t = Tuple[torch.Tensor]  # Input x
aten_op = "torch.ops.aten.asinh.default"

test_data_suite = {
    "zeros": lambda: torch.zeros(1, 5, 3, 2),
    "ones": lambda: torch.ones(10, 10, 10),
    "neg_ones": lambda: -torch.ones(10, 10, 10),
    "rand": lambda: (torch.rand(10, 10) - 0.5) * 20,
    "ramp": lambda: torch.linspace(-10.0, 10.0, steps=160),
    "near_zero": lambda: torch.tensor([-1e-6, 0.0, 1e-6]),
    "large": lambda: torch.tensor([-100.0, -10.0, 0.0, 10.0, 100.0]),
    "rand_4d": lambda: torch.randn(1, 3, 4, 5),
}


class Asinh(torch.nn.Module):
    def forward(self, x):
        return torch.asinh(x)


@common.parametrize("test_data", test_data_suite)
def test_asin_tosa_MI(test_data: Tuple):
    pipeline = TosaPipelineMI[input_t](
        Asinh(),
        (test_data(),),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_asin_tosa_BI(test_data: Tuple):
    pipeline = TosaPipelineBI[input_t](
        Asinh(),
        (test_data(),),
        aten_op=[],
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_asin_u55_BI(test_data: Tuple):
    pipeline = EthosU55PipelineBI[input_t](
        Asinh(),
        (test_data(),),
        aten_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_asin_u85_BI(test_data: Tuple):
    pipeline = EthosU85PipelineBI[input_t](
        Asinh(),
        (test_data(),),
        aten_ops=[],
    )
    pipeline.run()
