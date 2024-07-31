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

test_data = {
    "5d_float": (torch.FloatTensor([1, 2, 3, 5, 7]),),
    "1d_ones": ((3 * torch.ones(8),)),
    "1d_randn": (10 * torch.randn(8),),
    "4d_ones_1": (torch.ones(1, 1, 4, 4),),
    "4d_ones_2": (torch.ones(1, 3, 4, 2),),
}
T1 = Tuple[torch.Tensor]

test_data2 = {
    "5d_float": (
        torch.FloatTensor([1, 2, 3, 5, 7]),
        (torch.FloatTensor([2, 1, 2, 1, 10])),
    ),
    "4d_ones": (torch.ones(1, 10, 4, 6), torch.ones(1, 10, 4, 6)),
    "4d_randn_1": (torch.randn(1, 1, 4, 4), torch.ones(1, 1, 4, 1)),
    "4d_randn_2": (torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4)),
    "4d_randn_big": (10000 * torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 1)),
}
T2 = Tuple[torch.Tensor, torch.Tensor]


class Add(torch.nn.Module):
    def forward(self, x):
        return x + x


@common.parametrize("test_data", test_data)
def test_add_tosa_MI(test_data):
    pipeline = TosaPipelineMI[T1](Add(), test_data, aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", test_data)
def test_add_tosa_BI(test_data):
    pipeline = TosaPipelineBI[T1](Add(), test_data, aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", test_data)
def test_add_u55_BI(test_data):
    pipeline = EthosU55PipelineBI[T1](
        Add(), test_data, aten_op, exir_op, run_on_fvp=False
    )
    pipeline.run()


@common.parametrize("test_data", test_data)
def test_add_u85_BI(test_data):
    pipeline = EthosU85PipelineBI[T1](
        Add(), test_data, aten_op, exir_op, run_on_fvp=False
    )
    pipeline.run()


@common.parametrize("test_data", test_data)
@common.u55_fvp_mark
def test_add_u55_BI_on_fvp(test_data):
    pipeline = EthosU55PipelineBI[T1](
        Add(), test_data, aten_op, exir_op, run_on_fvp=True
    )
    pipeline.run()


@common.parametrize("test_data", test_data)
@common.u85_fvp_mark
def test_add_u85_BI_on_fvp(test_data):
    pipeline = EthosU85PipelineBI[T1](
        Add(), test_data, aten_op, exir_op, run_on_fvp=True
    )
    pipeline.run()


class Add2(torch.nn.Module):
    def forward(self, x, y):
        return x + y


@common.parametrize("test_data", test_data2)
def test_add2_tosa_MI(test_data):
    pipeline = TosaPipelineMI[T2](Add2(), test_data, aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", test_data2)
def test_add2_tosa_BI(test_data):
    pipeline = TosaPipelineBI[T2](Add2(), test_data, aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", test_data2)
def test_add2_u55_BI(test_data):
    pipeline = EthosU55PipelineBI[T2](
        Add2(), test_data, aten_op, exir_op, run_on_fvp=False
    )
    pipeline.run()


@common.parametrize("test_data", test_data2)
@common.u55_fvp_mark
def test_add2_u55_BI_on_fvp(test_data):
    pipeline = EthosU55PipelineBI[T2](
        Add2(), test_data, aten_op, exir_op, run_on_fvp=True
    )
    pipeline.run()


@common.parametrize("test_data", test_data2)
def test_add2_u85_BI(test_data):
    pipeline = EthosU85PipelineBI[T2](
        Add2(), test_data, aten_op, exir_op, run_on_fvp=False
    )
    pipeline.run()


@common.parametrize("test_data", test_data2)
@common.u85_fvp_mark
def test_add2_u85_BI_on_fvp(test_data):
    pipeline = EthosU85PipelineBI[T2](
        Add2(), test_data, aten_op, exir_op, run_on_fvp=True
    )
    pipeline.run()
