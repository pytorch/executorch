# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional, Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)


input_t = Tuple[torch.Tensor]


class Silu(torch.nn.Module):
    def forward(
        self,
        _input: torch.Tensor,
        _inplace: Optional[bool] = False,
    ):
        return torch.nn.SiLU(inplace=_inplace)(_input)

    test_data: list[input_t] = {
        "op_silu_rank1_ones": lambda: torch.ones(5),
        "op_silu_rank1_negative_ones": lambda: torch.ones(5) * (-1),
        "op_silu_rank1_rand": lambda: torch.rand(5) * 5,
        "op_silu_rank4_ones": lambda: torch.ones(1, 10, 25, 20),
        "op_silu_rank4_negative_ones": lambda: (-1) * torch.ones(1, 10, 25, 20),
        "op_silu_rank4_large_rand": lambda: 200 * torch.rand(1, 10, 25, 20),
        "op_silu_rank4_negative_large_rand": lambda: (-200) * torch.rand(1, 10, 25, 20),
        "op_silu_rank4_large_randn": lambda: 200 * torch.randn(1, 10, 25, 20) + 1,
    }

    aten_op_MI = "torch.ops.aten.silu.default"
    aten_op_inplace_MI = "torch.ops.aten.silu_.default"
    aten_op_BI = ["torch.ops.aten.sigmoid.default", "torch.ops.aten.mul.Tensor"]


@common.parametrize("test_data", Silu.test_data)
def test_silu_tosa_MI(test_data: input_t):
    silu_data = (test_data(), False)
    pipeline = TosaPipelineMI[input_t](Silu(), silu_data, Silu.aten_op_MI)
    pipeline.run()


@common.parametrize("test_data", Silu.test_data)
def test_silu_tosa_MI_inplace(test_data: input_t):
    silu_data = (test_data(), True)
    pipeline = TosaPipelineMI[input_t](Silu(), silu_data, Silu.aten_op_inplace_MI)
    pipeline.run()


@common.parametrize("test_data", Silu.test_data)
def test_silu_tosa_BI(test_data: input_t):
    silu_data = (test_data(), False)
    pipeline = TosaPipelineBI[input_t](Silu(), silu_data, Silu.aten_op_BI)
    pipeline.run()


@common.parametrize("test_data", Silu.test_data)
def test_silu_tosa_BI_inplace(test_data: input_t):
    silu_data = (test_data(), True)
    pipeline = TosaPipelineBI[input_t](Silu(), silu_data, Silu.aten_op_BI)
    pipeline.run()


@common.parametrize("test_data", Silu.test_data)
@common.XfailIfNoCorstone300
def test_silu_u55_BI(test_data: input_t):
    silu_data = (test_data(), False)
    pipeline = EthosU55PipelineBI[input_t](
        Silu(), silu_data, Silu.aten_op_BI, run_on_fvp=True
    )
    pipeline.run()


@common.parametrize("test_data", Silu.test_data)
@common.XfailIfNoCorstone300
def test_silu_u55_BI_inplace(test_data: input_t):
    silu_data = (test_data(), True)
    pipeline = EthosU55PipelineBI[input_t](
        Silu(), silu_data, Silu.aten_op_BI, run_on_fvp=True
    )
    pipeline.run()


@common.parametrize("test_data", Silu.test_data)
@common.XfailIfNoCorstone320
def test_silu_u85_BI(test_data: input_t):
    silu_data = (test_data(), False)
    pipeline = EthosU85PipelineBI[input_t](
        Silu(), silu_data, Silu.aten_op_BI, run_on_fvp=True
    )
    pipeline.run()


@common.parametrize("test_data", Silu.test_data)
@common.XfailIfNoCorstone320
def test_silu_u85_BI_inplace(test_data: input_t):
    silu_data = (test_data(), True)
    pipeline = EthosU85PipelineBI[input_t](
        Silu(), silu_data, Silu.aten_op_BI, run_on_fvp=True
    )
    pipeline.run()
