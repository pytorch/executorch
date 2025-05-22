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
        "zeros": lambda: (torch.zeros(1, 1280, 7, 7),),
        "ones": lambda: (torch.ones(1, 1280, 7, 7),),
        "rand": lambda: (torch.rand(1, 1280, 7, 7),),
        "randn": lambda: (torch.randn(1, 1280, 7, 7),),
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
        test_data(),
        AdaptiveAveragePool2d.aten_op,
        AdaptiveAveragePool2d.exir_op,
    ).run()


@common.parametrize("test_data", AdaptiveAveragePool2d.test_data_suite)
def test_adaptive_avg_pool2d_tosa_BI(test_data):
    TosaPipelineBI[input_t](
        AdaptiveAveragePool2d(),
        test_data(),
        AdaptiveAveragePool2d.aten_op,
        AdaptiveAveragePool2d.exir_op,
        symmetric_io_quantization=True,
    ).run()


@common.parametrize("test_data", AdaptiveAveragePool2d.test_data_suite)
@common.XfailIfNoCorstone300
def test_adaptive_avg_pool2d_u55_BI(test_data):
    EthosU55PipelineBI[input_t](
        AdaptiveAveragePool2d(),
        test_data(),
        AdaptiveAveragePool2d.aten_op,
        AdaptiveAveragePool2d.exir_op,
        run_on_fvp=True,
        symmetric_io_quantization=True,
    ).run()


@common.parametrize("test_data", AdaptiveAveragePool2d.test_data_suite)
@common.XfailIfNoCorstone320
def test_adaptive_avg_pool2d_u85_BI(test_data):
    EthosU85PipelineBI[input_t](
        AdaptiveAveragePool2d(),
        test_data(),
        AdaptiveAveragePool2d.aten_op,
        AdaptiveAveragePool2d.exir_op,
        run_on_fvp=True,
        symmetric_io_quantization=True,
    ).run()


class MeanDim(torch.nn.Module):
    test_data_suite: dict[str, tuple] = {
        "rank_1_keepdim": lambda: (
            torch.rand(7),
            (0),
            True,
        ),
        "rank_2_keepdim": lambda: (
            torch.rand(7, 7),
            (0, 1),
            True,
        ),
        "rank_3_keepdim": lambda: (
            torch.rand(7, 7, 7),
            (0, 1, 2),
            True,
        ),
        "rand_1_keepdim": lambda: (
            torch.rand(1, 7, 7, 7),
            (1),
            True,
        ),
        "rand_2_keepdim": lambda: (
            torch.rand(1, 7, 7, 7),
            (2),
            True,
        ),
        "rand_3_keepdim": lambda: (
            torch.rand(1, 7, 7, 7),
            (3),
            True,
        ),
        "rand_12_keepdim": lambda: (
            torch.rand(1, 7, 7, 7),
            (1, 2),
            True,
        ),
        "rand_13_keepdim": lambda: (
            torch.rand(1, 7, 7, 7),
            (1, 3),
            True,
        ),
        "rand_23_keepdim": lambda: (
            torch.rand(1, 7, 7, 7),
            (2, 3),
            True,
        ),
        "rand_123_keepdim": lambda: (
            torch.rand(1, 7, 7, 7),
            (1, 2, 3),
            True,
        ),
        "rand_0123_keepdim": lambda: (
            torch.rand(1, 7, 7, 7),
            (0, 1, 2, 3),
            True,
        ),
        "rank_1": lambda: (
            torch.rand(7),
            (-1),
            False,
        ),
        "rank_2": lambda: (
            torch.rand(7, 7),
            (-2, -1),
            False,
        ),
        "rank_3": lambda: (
            torch.rand(7, 7, 7),
            (-3, -2, -1),
            False,
        ),
        "rand_1": lambda: (
            torch.rand(1, 7, 7, 7),
            (-3),
            False,
        ),
        "rand_2": lambda: (
            torch.rand(1, 7, 7, 7),
            (-2),
            False,
        ),
        "rand_3": lambda: (
            torch.rand(1, 7, 7, 7),
            (-1),
            False,
        ),
        "rand_12": lambda: (
            torch.rand(1, 7, 7, 7),
            (-3, -2),
            False,
        ),
        "rand_13": lambda: (
            torch.rand(1, 7, 7, 7),
            (-3, -1),
            False,
        ),
        "rand_23": lambda: (
            torch.rand(1, 7, 7, 7),
            (-2, -1),
            False,
        ),
        "rand_123": lambda: (
            torch.rand(1, 7, 7, 7),
            (-3, -2, -1),
            False,
        ),
        "rand_0123": lambda: (
            torch.rand(1, 7, 7, 7),
            (-4, -3, -2, -1),
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
def test_mean_dim_tosa_MI(test_data):
    test_data, dim, keep_dim = test_data()
    TosaPipelineMI[input_t](
        MeanDim(dim, keep_dim),
        (test_data,),
        MeanDim.torch_op,
        MeanDim.exir_op,
    ).run()


@common.parametrize("test_data", MeanDim.test_data_suite)
def test_mean_dim_tosa_BI(test_data):
    test_data, dim, keep_dim = test_data()
    pipeline = TosaPipelineBI[input_t](
        MeanDim(dim, keep_dim),
        (test_data,),
        [],  # Might be sum, avgpool, or both
        symmetric_io_quantization=True,
    )
    pipeline.run()


@common.parametrize("test_data", MeanDim.test_data_suite)
@common.XfailIfNoCorstone300
def test_mean_dim_u55_BI(test_data):
    test_data, dim, keep_dim = test_data()
    pipeline = EthosU55PipelineBI[input_t](
        MeanDim(dim, keep_dim),
        (test_data,),
        [],  # Might be sum, avgpool, or both
        run_on_fvp=True,
        symmetric_io_quantization=True,
    ).dump_artifact("export")
    pipeline.run()


@common.parametrize("test_data", MeanDim.test_data_suite)
@common.XfailIfNoCorstone320
def test_mean_dim_u85_BI(test_data):
    test_data, dim, keep_dim = test_data()
    pipeline = EthosU85PipelineBI[input_t](
        MeanDim(dim, keep_dim),
        (test_data,),
        [],  # Might be sum, avgpool, or both
        run_on_fvp=True,
        symmetric_io_quantization=True,
    )
    pipeline.run()
