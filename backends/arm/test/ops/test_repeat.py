# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the repeat op which copies the data of the input tensor (possibly with new data format)
#


from typing import Sequence, Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)

input_t1 = Tuple[torch.Tensor, torch.Tensor]  # Input x, Input y
aten_op = "torch.ops.aten.repeat.default"


"""Tests Tensor.repeat for different ranks and dimensions."""


class Repeat(torch.nn.Module):
    # (input tensor, multiples)
    test_parameters = {
        "1_x_1": lambda: (torch.randn(3), (2,)),
        "2_x_2": lambda: (torch.randn(3, 4), (2, 1)),
        "4_x_4": lambda: (torch.randn(1, 1, 2, 2), (1, 2, 3, 4)),
        "1_x_2": lambda: (torch.randn(3), (2, 2)),
        "1_x_3": lambda: (torch.randn(3), (1, 2, 3)),
        "2_x_3": lambda: (torch.randn((3, 3)), (2, 2, 2)),
        "1_x_4": lambda: (torch.randn((3, 3, 3)), (2, 1, 2, 4)),
    }

    def forward(self, x: torch.Tensor, multiples: Sequence):
        return x.repeat(multiples)


@common.parametrize("test_data", Repeat.test_parameters)
def test_repeat_tosa_MI(test_data: Tuple):
    pipeline = TosaPipelineMI[input_t1](
        Repeat(),
        test_data(),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", Repeat.test_parameters)
def test_repeat_tosa_BI(test_data: Tuple):
    pipeline = TosaPipelineBI[input_t1](
        Repeat(),
        test_data(),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", Repeat.test_parameters)
def test_repeat_u55_BI(test_data: Tuple):
    pipeline = EthosU55PipelineBI[input_t1](
        Repeat(),
        test_data(),
        aten_op,
        exir_ops=[],
        run_on_fvp=False,
    )
    pipeline.run()


@common.parametrize("test_data", Repeat.test_parameters)
def test_repeat_u85_BI(test_data: Tuple):
    pipeline = EthosU85PipelineBI[input_t1](
        Repeat(),
        test_data(),
        aten_op,
        exir_ops=[],
        run_on_fvp=False,
    )
    pipeline.run()
