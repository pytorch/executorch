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


class Gelu(torch.nn.Module):
    aten_op = "torch.ops.aten.gelu.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_gelu_default"

    test_data: dict[str, Tuple[str, input_t1]] = {
        "zeros_none": lambda: (
            "none",
            torch.zeros(1, 10, 10, 10),
        ),
        "ones_none": lambda: (
            "none",
            torch.ones(10, 10, 10),
        ),
        "rand_none": lambda: (
            "none",
            (torch.rand(10, 10) - 0.5),
        ),
        "randn_pos_none": lambda: (
            "none",
            (torch.randn(1, 4, 4, 4) + 10),
        ),
        "randn_neg_none": lambda: (
            "none",
            (torch.randn(1, 4, 4, 4) - 10),
        ),
        "ramp_none": lambda: (
            "none",
            torch.arange(-16, 16, 0.2),
        ),
        "zeros_tanh": lambda: (
            "tanh",
            torch.zeros(1, 10, 10, 10),
        ),
        "ones_tanh": lambda: (
            "tanh",
            torch.ones(10, 10, 10),
        ),
        "rand_tanh": lambda: (
            "tanh",
            (torch.rand(10, 10) - 0.5),
        ),
        "randn_pos_tanh": lambda: (
            "tanh",
            (torch.randn(1, 4, 4, 4) + 10),
        ),
        "randn_neg_tanh": lambda: (
            "tanh",
            (torch.randn(1, 4, 4, 4) - 10),
        ),
        "ramp_tanh": lambda: (
            "tanh",
            torch.arange(-16, 16, 0.2),
        ),
    }

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.gelu = torch.nn.GELU(approximate)

    def forward(self, x: torch.Tensor):
        return self.gelu(x)


@common.parametrize("test_data", Gelu.test_data)
def test_gelu_tosa_FP(test_data: input_t1):
    approximate, test_data = test_data()
    TosaPipelineFP[input_t1](
        Gelu(approximate),
        (test_data,),
        Gelu.aten_op,
        Gelu.exir_op,
        use_to_edge_transform_and_lower=False,
    ).run()


@common.parametrize("test_data", Gelu.test_data)
def test_gelu_tosa_INT(test_data: input_t1):
    approximate, test_data = test_data()
    TosaPipelineINT[input_t1](
        Gelu(approximate),
        (test_data,),
        Gelu.aten_op,
        Gelu.exir_op,
    ).run()


@common.parametrize("test_data", Gelu.test_data)
@common.XfailIfNoCorstone300
def test_gelu_u55_INT(test_data: input_t1):
    approximate, test_data = test_data()
    EthosU55PipelineINT[input_t1](
        Gelu(approximate),
        (test_data,),
        Gelu.aten_op,
        Gelu.exir_op,
    ).run()


@common.parametrize("test_data", Gelu.test_data)
@common.XfailIfNoCorstone320
def test_gelu_u85_INT(test_data: input_t1):
    approximate, test_data = test_data()
    EthosU85PipelineINT[input_t1](
        Gelu(approximate),
        (test_data,),
        Gelu.aten_op,
        Gelu.exir_op,
    ).run()


@common.parametrize("test_data", Gelu.test_data)
@common.SkipIfNoModelConverter
def test_gelu_vgf_FP(test_data: input_t1):
    approximate, data = test_data()
    pipeline = VgfPipeline[input_t1](
        Gelu(approximate),
        (data,),
        Gelu.aten_op,
        Gelu.exir_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", Gelu.test_data)
@common.SkipIfNoModelConverter
def test_gelu_vgf_INT(test_data: input_t1):
    approximate, data = test_data()
    pipeline = VgfPipeline[input_t1](
        Gelu(approximate),
        (data,),
        Gelu.aten_op,
        Gelu.exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
