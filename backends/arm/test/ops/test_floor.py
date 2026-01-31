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

input_t1 = Tuple[torch.Tensor]


class Floor(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.floor(x)

    aten_op = "torch.ops.aten.floor.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_floor_default"


zeros = torch.zeros(1, 10, 10, 10)
ones = torch.ones(10, 10, 10)
rand = torch.rand(10, 10) - 0.5
randn_pos = torch.randn(1, 4, 4, 4) + 10
randn_neg = torch.randn(1, 4, 4, 4) - 10
ramp = torch.arange(-16, 16, 0.2)

test_data = {
    "floor_zeros": lambda: (Floor(), zeros),
    "floor_ones": lambda: (Floor(), ones),
    "floor_rand": lambda: (Floor(), rand),
    "floor_randn_pos": lambda: (Floor(), randn_pos),
    "floor_randn_neg": lambda: (Floor(), randn_neg),
    "floor_ramp": lambda: (Floor(), ramp),
}

test_data_bf16 = {
    "floor_rand_bf16": lambda: (
        Floor(),
        torch.rand(4, 4, dtype=torch.bfloat16) - 0.5,
    ),
    "floor_ramp_bf16": lambda: (
        Floor(),
        torch.arange(-8, 8, 0.5, dtype=torch.bfloat16),
    ),
}


@common.parametrize("test_data", test_data | test_data_bf16)
def test_floor_tosa_FP(test_data: input_t1):
    module, data = test_data()
    pipeline = TosaPipelineFP[input_t1](
        module,
        (data,),
        module.aten_op,
        module.exir_op,
        tosa_extensions=["bf16"],
    )
    pipeline.run()


@common.parametrize("test_data", test_data)
def test_floor_tosa_INT(test_data: input_t1):
    module, data = test_data()
    pipeline = TosaPipelineINT[input_t1](
        module,
        (data,),
        module.aten_op,
        module.exir_op,
        atol=0.06,
        rtol=0.01,
    )
    pipeline.run()


@common.parametrize("test_data", test_data)
@common.XfailIfNoCorstone300
def test_floor_u55_INT(test_data: input_t1):
    module, data = test_data()
    pipeline = EthosU55PipelineINT[input_t1](
        module,
        (data,),
        module.aten_op,
        module.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data)
@common.XfailIfNoCorstone320
def test_floor_u85_INT(test_data: input_t1):
    module, data = test_data()
    pipeline = EthosU85PipelineINT[input_t1](
        module,
        (data,),
        module.aten_op,
        module.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data)
@common.SkipIfNoModelConverter
def test_floor_vgf_no_quant(test_data: input_t1):
    module, data = test_data()
    pipeline = VgfPipeline[input_t1](
        module,
        (data,),
        module.aten_op,
        module.exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data)
@common.SkipIfNoModelConverter
def test_floor_vgf_quant(test_data: input_t1):
    module, data = test_data()
    pipeline = VgfPipeline[input_t1](
        module,
        (data,),
        module.aten_op,
        module.exir_op,
        atol=0.06,
        rtol=0.01,
        quantize=True,
    )
    pipeline.run()
