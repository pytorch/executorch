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


aten_op = "torch.ops.aten.floor.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_floor_default"

input_t1 = Tuple[torch.Tensor]  # Input x


class Floor(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.floor(x)

    test_data: dict[str, input_t1] = {
        "zeros": (torch.zeros(1, 10, 10, 10),),
        "ones": (torch.ones(10, 10, 10),),
        "rand": ((torch.rand(10, 10) - 0.5),),
        "randn_pos": ((torch.randn(1, 4, 4, 4) + 10),),
        "randn_neg": ((torch.randn(1, 4, 4, 4) - 10),),
        "ramp": (torch.arange(-16, 16, 0.2),),
    }


@common.parametrize("test_data", Floor.test_data)
def test_floor_tosa_MI(test_data: input_t1):
    pipeline = TosaPipelineMI[input_t1](Floor(), test_data, aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", Floor.test_data)
def test_floor_tosa_BI(test_data: input_t1):
    pipeline = TosaPipelineBI[input_t1](Floor(), test_data, aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", Floor.test_data)
def test_floor_u55_BI(test_data: input_t1):
    pipeline = EthosU55PipelineBI[input_t1](
        Floor(), test_data, aten_op, exir_op, run_on_fvp=False
    )
    pipeline.run()


@common.parametrize("test_data", Floor.test_data)
def test_floor_u85_BI(test_data: input_t1):
    pipeline = EthosU85PipelineBI[input_t1](
        Floor(), test_data, aten_op, exir_op, run_on_fvp=False
    )
    pipeline.run()


@common.parametrize("test_data", Floor.test_data)
@common.SkipIfNoCorstone300
def test_floor_u55_BI_on_fvp(test_data: input_t1):
    pipeline = EthosU55PipelineBI[input_t1](
        Floor(), test_data, aten_op, exir_op, run_on_fvp=True
    )
    pipeline.run()


@common.parametrize("test_data", Floor.test_data)
@common.SkipIfNoCorstone320
def test_floor_u85_BI_on_fvp(test_data: input_t1):
    pipeline = EthosU85PipelineBI[input_t1](
        Floor(), test_data, aten_op, exir_op, run_on_fvp=True
    )
    pipeline.run()
