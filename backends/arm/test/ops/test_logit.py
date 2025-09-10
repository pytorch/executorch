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

aten_op = "torch.ops.aten.logit.default"
exir_op = "executorch_exir_dialects_edge__ops_aten__logit_default"

input_t1 = Tuple[torch.Tensor]

test_data_suite = {
    "zeros": [torch.zeros((10, 10, 10)), None],
    "ones": [torch.ones((10, 10, 10)), None],
    "uniform_valid": [torch.rand((10, 10, 10)), None],
    "near_zero": [torch.full((10, 10), 1e-8), None],
    "near_one": [torch.full((10, 10), 1 - 1e-8), None],
    "mixed": [torch.tensor([0.0, 1e-5, 0.5, 1 - 1e-5, 1.0]), None],
    "multi_dim": [torch.rand((2, 3, 4)), None],
    "eps": [torch.zeros((10, 10, 10)), 1e-6],
    "invalid_neg": [torch.full((5,), -0.1), 1e-6],
    "invalid_gt1": [torch.full((5,), 1.1), 1e-6],
}


class Logit(torch.nn.Module):

    def forward(self, x: torch.Tensor, eps: torch.float32):
        return torch.logit(x, eps=eps)


@common.parametrize("test_data", test_data_suite)
def test_logit_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        Logit(),
        (*test_data,),
        aten_op=aten_op,
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_logit_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        Logit(),
        (*test_data,),
        aten_op=[],
        exir_op=exir_op,
    )
    pipeline.run()


@common.XfailIfNoCorstone300
@common.parametrize("test_data", test_data_suite)
def test_logit_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        Logit(),
        (*test_data,),
        aten_ops=[],
        exir_ops=exir_op,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("test_data", test_data_suite)
def test_logit_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        Logit(),
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
def test_logit_vgf_FP(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        Logit(),
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
def test_logit_vgf_INT(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        Logit(),
        (*test_data,),
        [],
        [],
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
