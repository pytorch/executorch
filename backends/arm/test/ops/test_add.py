# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
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

aten_op = "torch.ops.aten.add.Tensor"
exir_op = "executorch_exir_dialects_edge__ops_aten_add_Tensor"

input_t1 = Tuple[torch.Tensor]  # Input x


class Add(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x + x

    test_data: list[input_t1] = {
        "5d_float": (torch.FloatTensor([1, 2, 3, 5, 7]),),
        "1d_ones": ((3 * torch.ones(8),)),
        "1d_randn": (10 * torch.randn(8),),
        "4d_ones_1": (torch.ones(1, 1, 4, 4),),
        "4d_ones_2": (torch.ones(1, 3, 4, 2),),
    }


input_t2 = Tuple[torch.Tensor, torch.Tensor]  # Input x, y


class Add2(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x + y

    test_data: list[input_t2] = {
        "5d_float": (
            torch.FloatTensor([1, 2, 3, 5, 7]),
            (torch.FloatTensor([2, 1, 2, 1, 10])),
        ),
        "4d_ones": (torch.ones(1, 10, 4, 6), torch.ones(1, 10, 4, 6)),
        "4d_randn_1": (torch.randn(1, 1, 4, 4), torch.ones(1, 1, 4, 1)),
        "4d_randn_2": (torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4)),
        "4d_randn_big": (10000 * torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 1)),
    }


@common.parametrize("test_data", Add.test_data)
def test_add_tosa_MI(test_data: input_t1):
    pipeline = TosaPipelineMI[input_t1](Add(), test_data, aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", Add.test_data)
def test_add_tosa_BI(test_data: input_t1):
    pipeline = TosaPipelineBI[input_t1](Add(), test_data, aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", Add.test_data)
def test_add_u55_BI(test_data: input_t1):
    pipeline = EthosU55PipelineBI[input_t1](
        Add(), test_data, aten_op, exir_op, run_on_fvp=False
    )
    pipeline.run()


@common.parametrize("test_data", Add.test_data)
def test_add_u85_BI(test_data: input_t1):
    pipeline = EthosU85PipelineBI[input_t1](
        Add(), test_data, aten_op, exir_op, run_on_fvp=False
    )
    pipeline.run()


@common.parametrize("test_data", Add.test_data)
@common.SkipIfNoCorstone300
def test_add_u55_BI_on_fvp(test_data: input_t1):
    pipeline = EthosU55PipelineBI[input_t1](
        Add(), test_data, aten_op, exir_op, run_on_fvp=True
    )
    pipeline.run()


@common.parametrize("test_data", Add.test_data)
@common.SkipIfNoCorstone320
def test_add_u85_BI_on_fvp(test_data: input_t1):
    pipeline = EthosU85PipelineBI[input_t1](
        Add(), test_data, aten_op, exir_op, run_on_fvp=True
    )
    pipeline.run()


@common.parametrize("test_data", Add2.test_data)
def test_add2_tosa_MI(test_data: input_t2):
    pipeline = TosaPipelineMI[input_t2](Add2(), test_data, aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", Add2.test_data)
def test_add2_tosa_BI(test_data: input_t2):
    pipeline = TosaPipelineBI[input_t2](Add2(), test_data, aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", Add2.test_data)
def test_add2_u55_BI(test_data: input_t2):
    pipeline = EthosU55PipelineBI[input_t2](
        Add2(), test_data, aten_op, exir_op, run_on_fvp=False
    )
    pipeline.run()


@common.parametrize("test_data", Add2.test_data)
@common.SkipIfNoCorstone300
def test_add2_u55_BI_on_fvp(test_data: input_t2):
    pipeline = EthosU55PipelineBI[input_t2](
        Add2(), test_data, aten_op, exir_op, run_on_fvp=True
    )
    pipeline.run()


@common.parametrize("test_data", Add2.test_data)
def test_add2_u85_BI(test_data: input_t2):
    pipeline = EthosU85PipelineBI[input_t2](
        Add2(), test_data, aten_op, exir_op, run_on_fvp=False
    )
    pipeline.run()


@common.parametrize("test_data", Add2.test_data)
@common.SkipIfNoCorstone320
def test_add2_u85_BI_on_fvp(test_data: input_t2):
    pipeline = EthosU85PipelineBI[input_t2](
        Add2(), test_data, aten_op, exir_op, run_on_fvp=True
    )
    pipeline.run()
