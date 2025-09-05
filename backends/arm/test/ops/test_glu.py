# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn.functional as F
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op = "torch.ops.aten.glu.default"
exir_op = "executorch_exir_dialects_edge__ops_aten__glu_default"


input_t1 = Tuple[torch.Tensor]

test_data_suite = {
    "zeros": [torch.zeros(10, 10, 2), -1],
    "ones": [torch.ones(10, 10, 2), -1],
    "rand": [torch.rand(10, 10, 2) - 0.5, -1],
    "randn_pos": [torch.randn(10, 2) + 10, -1],
    "randn_neg": [torch.randn(10, 2) - 10, -1],
    "ramp": [torch.linspace(-16, 15.8, 160).reshape(-1, 2), -1],
    "zeros_custom_dim": [torch.zeros(7, 10, 5), 1],
    "rand_custom_dim": [torch.rand(10, 3, 3) - 0.5, 0],
}


class Glu(torch.nn.Module):

    def forward(self, a: torch.Tensor, dim: int) -> torch.Tensor:
        return F.glu(a, dim=dim)


@common.parametrize(
    "test_data",
    test_data_suite,
)
def test_glu_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        Glu(),
        (*test_data,),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    test_data_suite,
)
def test_glu_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        Glu(),
        (*test_data,),
        aten_op=[],
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    test_data_suite,
)
@common.XfailIfNoCorstone300
def test_glu_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        Glu(),
        (*test_data,),
        aten_ops=[],
        exir_ops=exir_op,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    test_data_suite,
)
@common.XfailIfNoCorstone320
def test_glu_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        Glu(),
        (*test_data,),
        aten_ops=[],
        exir_ops=exir_op,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    test_data_suite,
)
@common.SkipIfNoModelConverter
def test_glu_vgf_FP(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        Glu(),
        (*test_data,),
        [],
        [],
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    test_data_suite,
)
@common.SkipIfNoModelConverter
def test_glu_vgf_INT(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        Glu(),
        (*test_data,),
        [],
        [],
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
