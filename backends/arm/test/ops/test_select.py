# Copyright (c) Meta Platforms, Inc. and affiliates.
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
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

input_t1 = Tuple[torch.Tensor, int, int]

test_data_suite = {
    # (test_data, dim, index)
    "select3d_neg_1_dim_0_index": lambda: (torch.zeros(5, 3, 20), -1, 0),
    "select3d_0_dim_neg_1_index": lambda: (torch.rand(5, 3, 20), 0, -1),
    "select3d_0_dim_4_index": lambda: (torch.zeros(5, 3, 20), 0, 4),
    "select3d_0_dim_2_index": lambda: (torch.ones(10, 10, 10), 0, 2),
    "select4d_0_dim_2_index": lambda: (torch.rand(5, 3, 20, 2), 0, 2),
    "select2d_0_dim_0_index": lambda: (torch.rand(10, 10) - 0.5, 0, 0),
    "select1d_0_dim_1_index": lambda: (torch.randn(10) + 10, 0, 1),
    "select1d_0_dim_0_index": lambda: (torch.randn(10) - 10, 0, 2),
    "select3d_0_dim_1_index": lambda: (torch.arange(-16, 16, 0.2), 0, 1),
}

test_data_not_delegated = {
    "select3d_large_after_squeeze": lambda: (torch.rand(3, 64, 3, 49, 32), 0, 0),
}

aten_op_copy = "torch.ops.aten.select_copy.int"
aten_op_int = "torch.ops.aten.select.int"


class SelectCopy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dim: int, index: int):
        return torch.select_copy(x, dim=dim, index=index)


class SelectInt(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dim: int, index: int):
        return torch.select(x, dim=dim, index=index)


@common.parametrize("test_data", test_data_suite)
def test_select_int_tosa_FP_copy(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        SelectCopy(),
        test_data(),
        aten_op=aten_op_copy,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_select_int_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        SelectInt(),
        test_data(),
        aten_op=aten_op_int,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_select_int_tosa_INT_copy(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        SelectCopy(),
        test_data(),
        aten_op=aten_op_copy,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_select_int_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        SelectInt(),
        test_data(),
        aten_op=aten_op_int,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_select_int_u55_INT_copy(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        SelectCopy(),
        test_data(),
        aten_op_copy,
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_select_int_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        SelectInt(),
        test_data(),
        aten_op_int,
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_not_delegated)
def test_select_int_u55_INT_not_delegated(test_data: Tuple):
    pipeline = OpNotSupportedPipeline[input_t1](
        SelectInt(),
        test_data(),
        {aten_op_copy: 0},
        n_expected_delegates=0,
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_select_int_u85_INT_copy(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        SelectCopy(),
        test_data(),
        aten_op_copy,
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_select_int_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        SelectInt(),
        test_data(),
        aten_op_int,
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_select_int_vgf_FP_copy(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        SelectCopy(), test_data(), aten_op_copy, [], tosa_version="TOSA-1.0+FP"
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_select_int_vgf_FP(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        SelectInt(), test_data(), aten_op_int, [], tosa_version="TOSA-1.0+FP"
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_select_int_vgf_INT_copy(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        SelectCopy(),
        test_data(),
        aten_op_copy,
        [],
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_select_int_vgf_INT(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        SelectInt(),
        test_data(),
        aten_op_int,
        [],
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
