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


input_t1 = Tuple[torch.Tensor, torch.Tensor]  # Input x, y


class MultipleOutputsModule(torch.nn.Module):
    inputs: dict[str, input_t1] = {
        "randn": (torch.randn(10, 4, 5), torch.randn(10, 4, 5)),
    }

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return (x * y, x.sum(dim=-1, keepdim=True))


@common.parametrize("test_data", MultipleOutputsModule.inputs)
def test_tosa_MI_pipeline(test_data: input_t1):
    pipeline = TosaPipelineMI[input_t1](MultipleOutputsModule(), test_data, [], [])
    pipeline.run()


@common.parametrize("test_data", MultipleOutputsModule.inputs)
def test_tosa_BI_pipeline(test_data: input_t1):
    pipeline = TosaPipelineBI[input_t1](
        MultipleOutputsModule(), test_data, [], [], qtol=1
    )
    pipeline.run()


@common.parametrize("test_data", MultipleOutputsModule.inputs)
@common.XfailIfNoCorstone300
def test_U55_pipeline(test_data: input_t1):
    pipeline = EthosU55PipelineBI[input_t1](
        MultipleOutputsModule(), test_data, [], [], qtol=1
    )
    pipeline.run()


@common.parametrize("test_data", MultipleOutputsModule.inputs)
@common.XfailIfNoCorstone320
def test_U85_pipeline(test_data: input_t1):
    pipeline = EthosU85PipelineBI[input_t1](
        MultipleOutputsModule(), test_data, [], [], qtol=1
    )
    pipeline.run()
