# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Union

import torch
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

test_data_suite = {
    # (test_name, input, other)
    "op_floor_div_rank1_ones": lambda: (
        torch.ones(5),
        torch.ones(5),
    ),
    "op_floor_div_rank1_rand": lambda: (
        torch.rand(5) * 5,
        torch.rand(5) * 5,
    ),
    "op_floor_div_rank4_negative_ones": lambda: (
        (-1) * torch.ones(5, 10, 25, 20),
        torch.ones(5, 10, 25, 20),
    ),
    "op_floor_div_rank4_ones_div_negative": lambda: (
        torch.ones(5, 10, 25, 20),
        (-1) * torch.ones(5, 10, 25, 20),
    ),
    "op_floor_div_rank4_randn_mutltiple_broadcasts": lambda: (
        torch.randn(1, 4, 4, 1),
        torch.randn(1, 1, 4, 4),
    ),
    "op_floor_div_rank4_randn_scalar": lambda: (
        torch.randn(1, 4, 4, 1),
        2,
    ),
    "op_floor_div_rank4_large_rand": lambda: (
        200 * torch.rand(5, 10, 25, 20),
        torch.rand(5, 10, 25, 20),
    ),
}


class FloorDivide(torch.nn.Module):
    aten_op = "torch.ops.aten.floor_divide.default"
    aten_ops_int = ["aten.mul.Tensor", "aten.reciprocal.default", "aten.floor.default"]
    exir_op = "executorch_exir_dialects_edge__ops_aten_div_Tensor_mode"
    exir_ops_int = [
        "executorch_exir_dialects_edge__ops_aten_reciprocal_default",
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
        "executorch_exir_dialects_edge__ops_aten_floor_default",
    ]

    def forward(
        self,
        input_: Union[torch.Tensor, torch.types.Number],
        other_: Union[torch.Tensor, torch.types.Number],
    ):
        return torch.floor_divide(input=input_, other=other_)


input_t1 = Tuple[torch.Tensor, Union[torch.Tensor, int]]


@common.parametrize("test_data", test_data_suite)
def test_floor_divide_tosa_FP(test_data: input_t1):
    pipeline = TosaPipelineFP[input_t1](
        FloorDivide(),
        test_data(),
        FloorDivide.aten_op,
        FloorDivide.exir_op,
        use_to_edge_transform_and_lower=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_floor_divide_tosa_INT(test_data: input_t1):
    pipeline = TosaPipelineINT[input_t1](
        FloorDivide(),
        test_data(),
        aten_op=FloorDivide.aten_ops_int,
        exir_op=FloorDivide.exir_ops_int,
        use_to_edge_transform_and_lower=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_floor_divide_u55_INT(test_data: input_t1):
    pipeline = EthosU55PipelineINT[input_t1](
        FloorDivide(),
        test_data(),
        aten_ops=FloorDivide.aten_ops_int,
        exir_ops=[],
        run_on_fvp=True,
        use_to_edge_transform_and_lower=False,
    )
    pipeline.pop_stage("check_not.exir")
    pipeline.pop_stage("check_count.exir")
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_floor_divide_u85_INT(test_data: input_t1):
    pipeline = EthosU85PipelineINT[input_t1](
        FloorDivide(),
        test_data(),
        aten_ops=FloorDivide.aten_ops_int,
        exir_ops=FloorDivide.exir_ops_int,
        run_on_fvp=True,
        use_to_edge_transform_and_lower=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_floor_divide_vgf_FP(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        FloorDivide(),
        test_data(),
        FloorDivide.aten_op,
        FloorDivide.exir_op,
        tosa_version="TOSA-1.0+FP",
        use_to_edge_transform_and_lower=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_floor_divide_vgf_INT(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        FloorDivide(),
        test_data(),
        aten_op=FloorDivide.aten_ops_int,
        exir_op=FloorDivide.exir_ops_int,
        tosa_version="TOSA-1.0+INT",
        use_to_edge_transform_and_lower=False,
    )
    pipeline.run()
