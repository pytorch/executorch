# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
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
def test_adaptive_avg_pool2d_tosa_FP(test_data):
    TosaPipelineFP[input_t](
        AdaptiveAveragePool2d(),
        test_data(),
        AdaptiveAveragePool2d.aten_op,
        AdaptiveAveragePool2d.exir_op,
    ).run()


@common.parametrize("test_data", AdaptiveAveragePool2d.test_data_suite)
def test_adaptive_avg_pool2d_tosa_INT(test_data):
    TosaPipelineINT[input_t](
        AdaptiveAveragePool2d(),
        test_data(),
        AdaptiveAveragePool2d.aten_op,
        AdaptiveAveragePool2d.exir_op,
        symmetric_io_quantization=True,
    ).run()


@common.parametrize("test_data", AdaptiveAveragePool2d.test_data_suite)
@common.XfailIfNoCorstone300
def test_adaptive_avg_pool2d_u55_INT(test_data):
    EthosU55PipelineINT[input_t](
        AdaptiveAveragePool2d(),
        test_data(),
        AdaptiveAveragePool2d.aten_op,
        AdaptiveAveragePool2d.exir_op,
        symmetric_io_quantization=True,
    ).run()


@common.parametrize("test_data", AdaptiveAveragePool2d.test_data_suite)
@common.XfailIfNoCorstone320
def test_adaptive_avg_pool2d_u85_INT(test_data):
    EthosU85PipelineINT[input_t](
        AdaptiveAveragePool2d(),
        test_data(),
        AdaptiveAveragePool2d.aten_op,
        AdaptiveAveragePool2d.exir_op,
        symmetric_io_quantization=True,
    ).run()


@common.parametrize("test_data", AdaptiveAveragePool2d.test_data_suite)
@common.SkipIfNoModelConverter
def test_adaptive_avg_pool2d_vgf_FP(test_data):
    pipeline = VgfPipeline[input_t](
        AdaptiveAveragePool2d(),
        test_data(),
        AdaptiveAveragePool2d.aten_op,
        AdaptiveAveragePool2d.exir_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", AdaptiveAveragePool2d.test_data_suite)
@common.SkipIfNoModelConverter
def test_adaptive_avg_pool2d_vgf_INT(test_data):
    pipeline = VgfPipeline[input_t](
        AdaptiveAveragePool2d(),
        test_data(),
        AdaptiveAveragePool2d.aten_op,
        AdaptiveAveragePool2d.exir_op,
        symmetric_io_quantization=True,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


class MeanDim(torch.nn.Module):
    test_data_suite: dict[str, tuple] = {
        "rank_1_keepdim": lambda: (
            torch.rand(7),
            (0),
            True,
        ),
        "rank_2_keepdim": lambda: (
            torch.rand(7, 3),
            (0, 1),
            True,
        ),
        "rank_3_keepdim": lambda: (
            torch.rand(5, 7, 3),
            (0, 1, 2),
            True,
        ),
        "rand_1_keepdim": lambda: (
            torch.rand(1, 5, 7, 3),
            (1),
            True,
        ),
        "rand_2_keepdim": lambda: (
            torch.rand(1, 5, 7, 3),
            (2),
            True,
        ),
        "rand_3_keepdim": lambda: (
            torch.rand(1, 5, 7, 3),
            (3),
            True,
        ),
        "rand_12_keepdim": lambda: (
            torch.rand(1, 5, 7, 3),
            (1, 2),
            True,
        ),
        "rand_13_keepdim": lambda: (
            torch.rand(1, 5, 7, 3),
            (1, 3),
            True,
        ),
        "rand_23_keepdim": lambda: (
            torch.rand(1, 5, 7, 3),
            (2, 3),
            True,
        ),
        "rand_123_keepdim": lambda: (
            torch.rand(1, 5, 7, 3),
            (1, 2, 3),
            True,
        ),
        "rand_0123_keepdim": lambda: (
            torch.rand(1, 5, 7, 3),
            (0, 1, 2, 3),
            True,
        ),
        "rank_1": lambda: (
            torch.rand(7),
            (-1),
            False,
        ),
        "rank_2": lambda: (
            torch.rand(5, 7),
            (-2, -1),
            False,
        ),
        "rank_3": lambda: (
            torch.rand(5, 7, 3),
            (-3, -2, -1),
            False,
        ),
        "rand_1": lambda: (
            torch.rand(1, 5, 7, 3),
            (-3),
            False,
        ),
        "rand_2": lambda: (
            torch.rand(1, 5, 7, 3),
            (-2),
            False,
        ),
        "rand_3": lambda: (
            torch.rand(1, 5, 7, 3),
            (-1),
            False,
        ),
        "rand_12": lambda: (
            torch.rand(1, 5, 7, 3),
            (-3, -2),
            False,
        ),
        "rand_13": lambda: (
            torch.rand(1, 5, 7, 3),
            (-3, -1),
            False,
        ),
        "rand_23": lambda: (
            torch.rand(1, 5, 7, 3),
            (-2, -1),
            False,
        ),
        "rand_123": lambda: (
            torch.rand(1, 5, 7, 3),
            (-3, -2, -1),
            False,
        ),
        "rand_0123": lambda: (
            torch.rand(1, 5, 7, 3),
            (-4, -3, -2, -1),
            False,
        ),
        "rank5_01234": lambda: (
            torch.rand(1, 1, 7, 3, 2),
            (-5, -4, -3, -2, -1),
            False,
        ),
        "rank5_234": lambda: (
            torch.rand(1, 1, 7, 3, 2),
            (-3, -2, -1),
            False,
        ),
        "rank5_12": lambda: (
            torch.rand(1, 1, 7, 3, 2),
            (1, 2),
            False,
        ),
        "rank5_2": lambda: (
            torch.rand(1, 4, 7, 3, 2),
            (2),
            False,
        ),
        "u55_avg_pool_not_supported": lambda: (
            torch.rand(1, 1, 1, 257),
            (0, 1, 2, 3),
            True,
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
def test_mean_dim_tosa_FP(test_data):
    test_data, dim, keep_dim = test_data()
    TosaPipelineFP[input_t](
        MeanDim(dim, keep_dim),
        (test_data,),
        MeanDim.torch_op,
        MeanDim.exir_op,
    ).run()


@common.parametrize("test_data", MeanDim.test_data_suite)
def test_mean_dim_tosa_INT(test_data):
    test_data, dim, keep_dim = test_data()
    pipeline = TosaPipelineINT[input_t](
        MeanDim(dim, keep_dim),
        (test_data,),
        [],  # Might be sum, avgpool, or both
        symmetric_io_quantization=True,
        custom_path="MEANDIM",
    )
    pipeline.run()


xfails = {
    "rank5_01234": "Rank 5 graph input currently not supported in EthosUBackend (passes since CHW are all averaged over so data order does not matter in this case)",
    "rank5_234": "Rank 5 graph input currently not supported in EthosUBackend (passes since CHW are all averaged over so data order does not matter in this case)",
    "rank5_12": "Rank 5 graph input currently not supported in EthosUBackend",
    "rank5_2": "Rank 5 graph input currently not supported in EthosUBackend",
}


@common.parametrize("test_data", MeanDim.test_data_suite, xfails=xfails, strict=False)
@common.XfailIfNoCorstone300
def test_mean_dim_u55_INT(test_data):
    test_data, dim, keep_dim = test_data()
    pipeline = EthosU55PipelineINT[input_t](
        MeanDim(dim, keep_dim),
        (test_data,),
        [],  # Might be sum, avgpool, or both
        symmetric_io_quantization=True,
    )
    pipeline.add_stage_after(
        "export",
        pipeline.tester.check_not,
        ["torch.ops.aten.adaptive_avg_pool2d.default"],
        suffix="avg_pool",
    )
    pipeline.run()


@common.parametrize("test_data", MeanDim.test_data_suite, xfails=xfails, strict=False)
@common.XfailIfNoCorstone320
def test_mean_dim_u85_INT(test_data):
    test_data, dim, keep_dim = test_data()
    pipeline = EthosU85PipelineINT[input_t](
        MeanDim(dim, keep_dim),
        (test_data,),
        [],  # Might be sum, avgpool, or both
        symmetric_io_quantization=True,
    )
    pipeline.run()


@common.parametrize("test_data", MeanDim.test_data_suite)
@common.SkipIfNoModelConverter
def test_mean_dim_vgf_FP(test_data):
    test_data_val, dim, keep_dim = test_data()
    pipeline = VgfPipeline[input_t](
        MeanDim(dim, keep_dim),
        (test_data_val,),
        MeanDim.torch_op,
        MeanDim.exir_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", MeanDim.test_data_suite)
@common.SkipIfNoModelConverter
def test_mean_dim_vgf_INT(test_data):
    test_data_val, dim, keep_dim = test_data()
    pipeline = VgfPipeline[input_t](
        MeanDim(dim, keep_dim),
        (test_data_val,),
        [],  # Might be sum, avgpool, or both
        symmetric_io_quantization=True,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
