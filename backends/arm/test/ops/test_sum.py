# Copyright 2024-2025 Arm Limited and/or its affiliates.
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
    }

    def forward(self, x: torch.Tensor, dim: int, keepdim: bool):
        return x.sum(dim=dim, keepdim=keepdim)


@common.parametrize("test_data", Sum.test_parameters)
def test_sum_dim_intlist_tosa_MI(test_data: input_t1):
    pipeline = TosaPipelineMI[input_t1](
        Sum(),
        test_data(),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", Sum.test_parameters)
def test_sum_dim_intlist_tosa_BI(test_data: input_t1):
    pipeline = TosaPipelineBI[input_t1](
        Sum(),
        test_data(),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", Sum.test_parameters)
@common.XfailIfNoCorstone300
def test_view_u55_BI_1_0(test_data: Tuple):
    pipeline = EthosU55PipelineBI[input_t1](
        Sum(),
        test_data(),
        aten_op,
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", Sum.test_parameters)
@common.XfailIfNoCorstone320
def test_view_u85_BI_1_0(test_data: Tuple):
    pipeline = EthosU85PipelineBI[input_t1](
        Sum(),
        test_data(),
        aten_op,
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


reject_inputs = {
    "reject_large_0_dim": lambda: (torch.rand((65537, 1, 1)), 0, False),
    "reject_large_2_dim": lambda: (torch.rand((800, 90, 1)), 2, False),
    "reject_large_1_dim": lambda: (torch.rand((3, 2, 800, 90)), 1, False),
}


@common.parametrize("test_data", reject_inputs)
def test_view_u55_BI_not_delegated(test_data: Tuple):
    pipeline = EthosU55PipelineBI[input_t1](
        Sum(),
        test_data(),
        aten_op,
        exir_ops=[],
        run_on_fvp=False,  # Run fails since we are missing a non partitioned sum op
    )
    pipeline.pop_stage("check_count.exir")
    pipeline.run()
