# Copyright 2024-2025 Arm Limited and/or its affiliates.
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

input_t1 = Tuple[torch.Tensor]  # Input x, Input y
aten_op = "torch.ops.aten.reciprocal.default"

test_data_suite = {
    "op_reciprocal_rank1_ones": lambda: torch.ones(5),
    "op_reciprocal_rank1_rand": lambda: torch.rand(5) * 5,
    "op_reciprocal_rank1_negative_ones": lambda: torch.ones(5) * (-1),
    "op_reciprocal_rank4_ones": lambda: torch.ones(1, 10, 25, 20),
    "op_reciprocal_rank4_negative_ones": lambda: (-1) * torch.ones(1, 10, 25, 20),
    "op_reciprocal_rank4_ones_reciprocal_negative": lambda: torch.ones(1, 10, 25, 20),
    "op_reciprocal_rank4_large_rand": lambda: 200 * torch.rand(1, 10, 25, 20),
    "op_reciprocal_rank4_negative_large_rand": lambda: (-200)
    * torch.rand(1, 10, 25, 20),
    "op_reciprocal_rank4_large_randn": lambda: 200 * torch.randn(1, 10, 25, 20) + 1,
}


class Reciprocal(torch.nn.Module):

    def forward(self, input_: torch.Tensor):
        return input_.reciprocal()


@common.parametrize("test_data", test_data_suite)
def test_reciprocal_tosa_FP(test_data: torch.Tensor):
    pipeline = TosaPipelineFP[input_t1](
        Reciprocal(),
        (test_data(),),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_reciprocal_tosa_INT(test_data: torch.Tensor):
    pipeline = TosaPipelineINT[input_t1](
        Reciprocal(),
        (test_data(),),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_reciprocal_u55_INT(test_data: torch.Tensor):
    pipeline = EthosU55PipelineINT[input_t1](
        Reciprocal(),
        (test_data(),),
        aten_op,
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_reciprocal_u85_INT(test_data: torch.Tensor):
    pipeline = EthosU85PipelineINT[input_t1](
        Reciprocal(),
        (test_data(),),
        aten_op,
        exir_ops=[],
        symmetric_io_quantization=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_reciprocal_vgf_no_quant(test_data: torch.Tensor):
    pipeline = VgfPipeline[input_t1](
        Reciprocal(),
        (test_data(),),
        aten_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_reciprocal_vgf_quant(test_data: torch.Tensor):
    pipeline = VgfPipeline[input_t1](
        Reciprocal(),
        (test_data(),),
        aten_op,
        quantize=True,
    )
    pipeline.run()
