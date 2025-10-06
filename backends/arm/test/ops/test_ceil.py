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

input_t1 = Tuple[torch.Tensor]


class Ceil(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.ceil(x)

    aten_op = "torch.ops.aten.ceil.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_ceil_default"


zeros = torch.zeros(1, 10, 10, 10)
ones = torch.ones(10, 10, 10)
rand = torch.rand(10, 10) - 0.5
randn_pos = torch.randn(1, 4, 4, 4) + 10
randn_neg = torch.randn(1, 4, 4, 4) - 10
ramp = torch.arange(-16, 16, 0.2)

test_data = {
    "ceil_zeros": lambda: (Ceil(), zeros),
    "ceil_ones": lambda: (Ceil(), ones),
    "ceil_rand": lambda: (Ceil(), rand),
    "ceil_randn_pos": lambda: (Ceil(), randn_pos),
    "ceil_randn_neg": lambda: (Ceil(), randn_neg),
    "ceil_ramp": lambda: (Ceil(), ramp),
}


@common.parametrize("test_data", test_data)
def test_ceil_tosa_FP(test_data: input_t1):
    module, data = test_data()
    pipeline = TosaPipelineFP[input_t1](
        module,
        (data,),
        module.aten_op,
        module.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data)
def test_ceil_tosa_INT(test_data: input_t1):
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
def test_ceil_u55_INT(test_data: input_t1):
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
def test_ceil_u85_INT(test_data: input_t1):
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
def test_ceil_vgf_FP(test_data: input_t1):
    module, data = test_data()
    pipeline = VgfPipeline[input_t1](
        module,
        (data,),
        module.aten_op,
        module.exir_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", test_data)
@common.SkipIfNoModelConverter
def test_ceil_vgf_INT(test_data: input_t1):
    module, data = test_data()
    pipeline = VgfPipeline[input_t1](
        module,
        (data,),
        module.aten_op,
        module.exir_op,
        atol=0.06,
        rtol=0.01,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
