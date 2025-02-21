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


input_t1 = Tuple[torch.Tensor]  # Input x


class Ceil(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.ceil(x)

    op_name = "ceil"
    aten_op = "torch.ops.aten.ceil.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_ceil_default"


class Floor(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.floor(x)

    op_name = "floor"
    aten_op = "torch.ops.aten.floor.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_floor_default"


zeros = torch.zeros(1, 10, 10, 10)
ones = torch.ones(10, 10, 10)
rand = torch.rand(10, 10) - 0.5
randn_pos = torch.randn(1, 4, 4, 4) + 10
randn_neg = torch.randn(1, 4, 4, 4) - 10
ramp = torch.arange(-16, 16, 0.2)


test_data = {
    "ceil_zeros": (
        Ceil(),
        zeros,
    ),
    "floor_zeros": (
        Floor(),
        zeros,
    ),
    "ceil_ones": (
        Ceil(),
        ones,
    ),
    "floor_ones": (
        Floor(),
        ones,
    ),
    "ceil_rand": (
        Ceil(),
        rand,
    ),
    "floor_rand": (
        Floor(),
        rand,
    ),
    "ceil_randn_pos": (
        Ceil(),
        randn_pos,
    ),
    "floor_randn_pos": (
        Floor(),
        randn_pos,
    ),
    "ceil_randn_neg": (
        Ceil(),
        randn_neg,
    ),
    "floor_randn_neg": (
        Floor(),
        randn_neg,
    ),
    "ceil_ramp": (
        Ceil(),
        ramp,
    ),
    "floor_ramp": (
        Floor(),
        ramp,
    ),
}


@common.parametrize("test_data", test_data)
def test_unary_tosa_MI(test_data: input_t1):
    module = test_data[0]
    pipeline = TosaPipelineMI[input_t1](
        module, (test_data[1],), module.aten_op, module.exir_op
    )
    pipeline.run()


@common.parametrize("test_data", test_data)
def test_unary_tosa_BI(test_data: input_t1):
    module = test_data[0]
    pipeline = TosaPipelineBI[input_t1](
        module, (test_data[1],), module.aten_op, module.exir_op
    )
    pipeline.run()


@common.parametrize("test_data", test_data)
def test_unary_u55_BI(test_data: input_t1):
    module = test_data[0]
    pipeline = EthosU55PipelineBI[input_t1](
        module, (test_data[1],), module.aten_op, module.exir_op, run_on_fvp=False
    )
    pipeline.run()


@common.parametrize("test_data", test_data)
def test_unary_u85_BI(test_data: input_t1):
    module = test_data[0]
    pipeline = EthosU85PipelineBI[input_t1](
        module, (test_data[1],), module.aten_op, module.exir_op, run_on_fvp=False
    )
    pipeline.run()


@common.parametrize("test_data", test_data)
@common.SkipIfNoCorstone300
def test_unary_u55_BI_on_fvp(test_data: input_t1):
    module = test_data[0]
    pipeline = EthosU55PipelineBI[input_t1](
        module, (test_data[1],), module.aten_op, module.exir_op, run_on_fvp=True
    )
    pipeline.run()


@common.parametrize("test_data", test_data)
@common.SkipIfNoCorstone320
def test_unary_u85_BI_on_fvp(test_data: input_t1):
    module = test_data[0]
    pipeline = EthosU85PipelineBI[input_t1](
        module, (test_data[1],), module.aten_op, module.exir_op, run_on_fvp=True
    )
    pipeline.run()
