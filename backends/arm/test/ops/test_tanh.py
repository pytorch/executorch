# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
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

aten_op = "torch.ops.aten.tanh.default"
input_t1 = Tuple[torch.Tensor]  # Input x

test_data_suite = {
    # (test_name, test_data)
    "zeros": lambda: torch.zeros(10, 10, 10, 10),
    "ones": lambda: torch.ones(10, 10, 10),
    "rand": lambda: torch.rand(10, 10) - 0.5,
    "randn_pos": lambda: torch.randn(10) + 10,
    "randn_neg": lambda: torch.randn(10) - 10,
    "ramp": lambda: torch.arange(-16, 16, 0.2),
}


class Tanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        return self.tanh(x)


@common.parametrize("test_data", test_data_suite)
def test_tanh_tosa_MI(test_data: Tuple):
    pipeline = TosaPipelineMI[input_t1](
        Tanh(),
        (test_data(),),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_tanh_tosa_BI(test_data: Tuple):
    pipeline = TosaPipelineBI[input_t1](
        Tanh(),
        (test_data(),),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_tanh_u55_BI(test_data: Tuple):
    pipeline = EthosU55PipelineBI[input_t1](
        Tanh(),
        (test_data(),),
        aten_op,
        exir_ops=[],
        run_on_fvp=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_tanh_u85_BI(test_data: Tuple):
    pipeline = EthosU85PipelineBI[input_t1](
        Tanh(),
        (test_data(),),
        aten_op,
        exir_ops=[],
        run_on_fvp=False,
    )
    pipeline.run()
