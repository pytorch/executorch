# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Tuple

import torch
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op = "torch.ops.aten.sum.dim_IntList"
input_t1 = Tuple[torch.Tensor]  # Input x


"""Tests sum which sums all elements along some specified dimensions.
keepdim specifies whether the dimension that is summed should
be squeezed or not.
"""


class Sum(torch.nn.Module):
    test_parameters = {
        "1d_dim_0_keep": lambda: (torch.rand(10), 0, True),
        "2d_dim_1_no_keep": lambda: (torch.rand(10, 10), 1, False),
        "3d_dims_keep": lambda: (torch.rand(10, 10, 10), [-3, 1], True),
        "4d_dims_no_keep": lambda: (torch.rand(1, 1, 5, 8), 1, False),
        "4d_dim_3_keep": lambda: (torch.rand(1, 2, 3, 4), 3, True),
        "4d_dims_keep": lambda: (torch.rand(1, 2, 8, 8), [2, 3, 0], True),
        "dim_None": lambda: (torch.rand(10), None, True),
        "dim_None_4d_tensor": lambda: (torch.rand(10, 3, 2, 1), None, True),
    }
    test_parameters_bf16 = {
        "1d_dim_0_keep_bf16": lambda: (torch.rand(12, dtype=torch.bfloat16), 0, True),
        "3d_dims_keep_bf16": lambda: (
            torch.rand(4, 6, 3, dtype=torch.bfloat16),
            [0, -1],
            True,
        ),
        "dim_None_bf16": lambda: (torch.rand(6, 2, dtype=torch.bfloat16), None, False),
    }

    def forward(self, x: torch.Tensor, dim: int, keepdim: bool):
        return x.sum(dim=dim, keepdim=keepdim)


@common.parametrize("test_data", Sum.test_parameters | Sum.test_parameters_bf16)
def test_sum_dim_intlist_tosa_FP(test_data: input_t1):
    test_data = test_data()
    match test_data[0].dtype:
        case torch.bfloat16:
            rtol = 1e-2
        case _:
            rtol = 1e-3

    pipeline = TosaPipelineFP[input_t1](
        Sum(),
        test_data,
        aten_op,
        exir_op=[],
        tosa_extensions=["bf16"],
        rtol=rtol,
    )
    pipeline.run()


@common.parametrize("test_data", Sum.test_parameters)
def test_sum_dim_intlist_tosa_INT(test_data: input_t1):
    pipeline = TosaPipelineINT[input_t1](
        Sum(),
        test_data(),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", Sum.test_parameters)
@common.XfailIfNoCorstone300
def test_view_u55_INT_1_0(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        Sum(),
        test_data(),
        aten_op,
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", Sum.test_parameters)
@common.XfailIfNoCorstone320
def test_view_u85_INT_1_0(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        Sum(),
        test_data(),
        aten_op,
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", Sum.test_parameters)
@common.SkipIfNoModelConverter
def test_sum_dim_intlist_vgf_no_quant(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        Sum(),
        test_data(),
        aten_op,
        run_on_vulkan_runtime=True,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", Sum.test_parameters)
@common.SkipIfNoModelConverter
def test_sum_dim_intlist_vgf_quant(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        Sum(),
        test_data(),
        aten_op,
        run_on_vulkan_runtime=True,
        quantize=True,
    )
    pipeline.run()


reject_inputs = {
    "reject_large_0_dim": lambda: (torch.rand((65537, 1, 1)), 0, False),
    "reject_large_2_dim": lambda: (torch.rand((800, 90, 1)), 2, False),
    "reject_large_1_dim": lambda: (torch.rand((3, 2, 800, 90)), 1, False),
}


@common.parametrize("test_data", reject_inputs)
def test_view_u55_INT_failure_set(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        Sum(),
        test_data(),
        aten_op,
        exir_ops=[],
        run_on_fvp=False,  # Run fails since we are missing a non partitioned sum op
    )
    pipeline.pop_stage("check_count.exir")
    pipeline.run()


input_t2 = tuple[torch.Tensor]


class SumDefault(torch.nn.Module):
    test_parameters = {
        "rank1": lambda: (torch.rand(10),),
        "rank2": lambda: (torch.rand(10, 1, 10),),
        "rank4": lambda: (torch.rand(1, 1, 5, 8),),
    }
    test_parameters_bf16 = {
        "rank1_bf16": lambda: (torch.rand(8, dtype=torch.bfloat16),),
        "rank3_bf16": lambda: (torch.rand(4, 3, 2, dtype=torch.bfloat16),),
    }
    aten_op = "torch.ops.aten.sum.default"

    def forward(self, x: torch.Tensor):
        return x.sum()


@common.parametrize(
    "test_data", SumDefault.test_parameters | SumDefault.test_parameters_bf16
)
def test_sum_tosa_FP(test_data: Callable[[], input_t2]):
    test_vector = test_data()
    match test_vector[0].dtype:
        case torch.bfloat16:
            atol = 5e-2
            rtol = 5e-2
        case _:
            atol = 1e-3
            rtol = 1e-3

    pipeline = TosaPipelineFP[input_t2](
        SumDefault(),
        test_vector,
        SumDefault.aten_op,
        tosa_extensions=["bf16"],
        atol=atol,
        rtol=rtol,
    )
    pipeline.run()


@common.parametrize("test_data", SumDefault.test_parameters)
def test_sum_tosa_INT(test_data: Callable[[], input_t2]):
    pipeline = TosaPipelineINT[input_t1](SumDefault(), test_data(), SumDefault.aten_op)
    pipeline.run()
