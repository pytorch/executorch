# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
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

input_t1 = Tuple[torch.Tensor]  # Input x

aten_op = "torch.ops.aten.relu.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_relu_default"

test_data_suite = {
    # (test_name, test_data)
    "zeros": lambda: torch.zeros(1, 10, 10, 10),
    "ones": lambda: torch.ones(10, 10, 10),
    "rand": lambda: torch.rand(10, 10) - 0.5,
    "randn_pos": lambda: torch.randn(10) + 10,
    "randn_neg": lambda: torch.randn(10) - 10,
    "ramp": lambda: torch.arange(-16, 16, 0.2),
}


class Relu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(x)


@common.parametrize("test_data", test_data_suite)
def test_relu_tosa_MI(test_data: torch.Tensor):
    pipeline = TosaPipelineMI[input_t1](
        Relu(),
        (test_data(),),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_relu_tosa_BI(test_data: torch.Tensor):
    pipeline = TosaPipelineBI[input_t1](
        Relu(),
        (test_data(),),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_relu_u55_BI(test_data: torch.Tensor):
    pipeline = EthosU55PipelineBI[input_t1](
        Relu(),
        (test_data(),),
        aten_op,
        exir_op,
        run_on_fvp=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_relu_u85_BI(test_data: torch.Tensor):
    pipeline = EthosU85PipelineBI[input_t1](
        Relu(),
        (test_data(),),
        aten_op,
        exir_op,
        run_on_fvp=False,
    )
    pipeline.run()
