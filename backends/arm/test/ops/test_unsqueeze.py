# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the unsqueeze op which copies the data of the input tensor (possibly with new data format)
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

aten_op = "torch.ops.aten.unsqueeze.default"
input_t1 = Tuple[torch.Tensor, torch.scalar_tensor]  # Input x, Input y


class Unsqueeze(torch.nn.Module):
    shapes: list[int | Sequence[int]] = [5, (5, 5), (5, 4), (5, 4, 3)]
    test_parameters = {}
    for n in shapes:
        test_parameters[f"rand_{n}"] = (torch.randn(n),)

    def forward(self, x: torch.Tensor, dim):
        return x.unsqueeze(dim)


@common.parametrize("test_tensor", Unsqueeze.test_parameters)
def test_unsqueeze_tosa_MI(test_tensor: torch.Tensor):
    for i in range(-test_tensor[0].dim() - 1, test_tensor[0].dim() + 1):
        pipeline = TosaPipelineMI[input_t1](
            Unsqueeze(),
            (*test_tensor, i),
            aten_op,
            exir_op=[],
        )
        pipeline.run()


@common.parametrize("test_tensor", Unsqueeze.test_parameters)
def test_unsqueeze_tosa_BI(test_tensor: torch.Tensor):
    pipeline = TosaPipelineBI[input_t1](
        Unsqueeze(),
        (*test_tensor, 0),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_tensor", Unsqueeze.test_parameters)
@common.XfailIfNoCorstone300
def test_unsqueeze_u55_BI(test_tensor: torch.Tensor):
    pipeline = EthosU55PipelineBI[input_t1](
        Unsqueeze(),
        (*test_tensor, 0),
        aten_op,
        exir_ops=[],
        run_on_fvp=False,
    )
    pipeline.run()


@common.parametrize("test_tensor", Unsqueeze.test_parameters)
@common.XfailIfNoCorstone320
def test_unsqueeze_u85_BI(test_tensor: torch.Tensor):
    pipeline = EthosU85PipelineBI[input_t1](
        Unsqueeze(),
        (*test_tensor, 0),
        aten_op,
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()
