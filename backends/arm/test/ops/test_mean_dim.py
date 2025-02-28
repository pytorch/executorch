# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)

input_t = tuple[torch.Tensor]


class AdaptiveAveragePool2d(torch.nn.Module):
    test_data_suite = {
        # (test_name, test_data)
        "zeros": (torch.zeros(1, 1280, 7, 7),),
        "ones": (torch.ones(1, 1280, 7, 7),),
        "rand": (torch.rand(1, 1280, 7, 7),),
        "randn": (torch.randn(1, 1280, 7, 7),),
    }
    aten_op = "torch.ops.aten.adaptive_avg_pool2d.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_mean_dim"

    def __init__(self):
        super().__init__()
        self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        return self.adaptive_avg_pool2d(x)


@common.parametrize("test_data", AdaptiveAveragePool2d.test_data_suite)
def test_adaptive_avg_pool2d_tosa_MI(test_data):
    TosaPipelineMI[input_t](
        AdaptiveAveragePool2d(),
        test_data,
        AdaptiveAveragePool2d.aten_op,
        AdaptiveAveragePool2d.exir_op,
    ).run()


@common.parametrize("test_data", AdaptiveAveragePool2d.test_data_suite)
def test_adaptive_avg_pool2d_tosa_BI(test_data):
    TosaPipelineBI[input_t](
        AdaptiveAveragePool2d(),
        test_data,
        AdaptiveAveragePool2d.aten_op,
        AdaptiveAveragePool2d.exir_op,
    ).run()


@common.parametrize("test_data", AdaptiveAveragePool2d.test_data_suite)
def test_adaptive_avg_pool2d_u55(test_data):
    EthosU55PipelineBI[input_t](
        AdaptiveAveragePool2d(),
        test_data,
        AdaptiveAveragePool2d.aten_op,
        AdaptiveAveragePool2d.exir_op,
    ).run()


@common.parametrize("test_data", AdaptiveAveragePool2d.test_data_suite)
def test_adaptive_avg_pool2d_u85(test_data):
    EthosU85PipelineBI[input_t](
        AdaptiveAveragePool2d(),
        test_data,
        AdaptiveAveragePool2d.aten_op,
        AdaptiveAveragePool2d.exir_op,
    ).run()


@common.parametrize("test_data", AdaptiveAveragePool2d.test_data_suite)
@common.SkipIfNoCorstone300
def test_adaptive_avg_pool2d_u55_on_fvp(test_data):
    EthosU55PipelineBI[input_t](
        AdaptiveAveragePool2d(),
        test_data,
        AdaptiveAveragePool2d.aten_op,
        AdaptiveAveragePool2d.exir_op,
        run_on_fvp=True,
    ).run()


@common.parametrize("test_data", AdaptiveAveragePool2d.test_data_suite)
@common.SkipIfNoCorstone320
def test_adaptive_avg_pool2d_u85_on_fvp(test_data):
    EthosU85PipelineBI[input_t](
        AdaptiveAveragePool2d(),
        test_data,
        AdaptiveAveragePool2d.aten_op,
        AdaptiveAveragePool2d.exir_op,
        run_on_fvp=True,
    ).run()


class MeanDim(torch.nn.Module):
    test_data_suite: dict[str, tuple] = {
        "zeros": (torch.zeros(1, 1280, 7, 7), -1, True),
        "ones": (torch.ones(1, 1280, 7, 7), (-1, 2), False),
        "rand": (
            torch.rand(1, 1280, 7, 7),
            (-1),
            True,
        ),
        "randn": (
            torch.randn(1, 1280, 7, 7),
            (-1, -2, -3),
            False,
        ),
    }
    torch_op = "torch.ops.aten.mean.dim"
    exir_op = "executorch_exir_dialects_edge__ops_aten_mean_dim"

    def __init__(self, dim: int | list[int] = -1, keepdim: bool = True):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor):
        return x.mean(dim=self.dim, keepdim=self.keepdim)


@common.parametrize("test_data", MeanDim.test_data_suite)
def test_mean_tosa_MI(test_data):
    TosaPipelineMI[input_t](
        MeanDim(test_data[1], test_data[2]),
        (test_data[0],),
        MeanDim.torch_op,
        MeanDim.exir_op,
    ).run()


@common.parametrize("test_data", MeanDim.test_data_suite)
def test_mean_tosa_BI(test_data):
    TosaPipelineBI[input_t](
        MeanDim(test_data[1], test_data[2]),
        (test_data[0],),
        "torch.ops.aten.sum.dim_IntList",  # Just check for sum op included in the mean decomposition
    ).run()


@common.parametrize("test_data", MeanDim.test_data_suite)
def test_mean_u55(test_data):
    EthosU55PipelineBI[input_t](
        MeanDim(test_data[1], test_data[2]),
        (test_data[0],),
        "torch.ops.aten.sum.dim_IntList",  # Just check for sum op included in the mean decomposition
    ).run()


@common.parametrize("test_data", MeanDim.test_data_suite)
def test_mean_u85(test_data):
    EthosU85PipelineBI[input_t](
        MeanDim(test_data[1], test_data[2]),
        (test_data[0],),
        "torch.ops.aten.sum.dim_IntList",  # Just check for sum op included in the mean decomposition
    ).run()


@common.parametrize("test_data", MeanDim.test_data_suite)
@common.SkipIfNoCorstone300
def test_mean_u55_on_fvp(test_data):
    EthosU55PipelineBI[input_t](
        MeanDim(test_data[1], test_data[2]),
        (test_data[0],),
        "torch.ops.aten.sum.dim_IntList",  # Just check for sum op included in the mean decomposition
        run_on_fvp=True,
    ).run()


@common.parametrize("test_data", MeanDim.test_data_suite)
@common.SkipIfNoCorstone320
def test_mean_u85_on_fvp(test_data):
    EthosU85PipelineBI[input_t](
        MeanDim(test_data[1], test_data[2]),
        (test_data[0],),
        "torch.ops.aten.sum.dim_IntList",  # Just check for sum op included in the mean decomposition
        run_on_fvp=True,
    ).run()
