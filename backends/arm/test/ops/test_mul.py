# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

input_t1 = Tuple[torch.Tensor, torch.Tensor]  # Input x
aten_op = "torch.ops.aten.mul.Tensor"

test_data_suite = {
    # (test_name, input, other,) See torch.mul() for info
    "op_mul_rank1_rand": lambda: (
        torch.rand(5) * 3.7,
        torch.rand(5) * 1.5,
    ),
    "op_mul_rank2_rand": lambda: (
        torch.rand(4, 5),
        torch.rand(1, 5),
    ),
    "op_mul_rank3_randn": lambda: (
        torch.randn(10, 5, 2),
        torch.randn(10, 5, 2),
    ),
    "op_mul_rank4_randn": lambda: (
        torch.randn(1, 10, 25, 20),
        torch.randn(1, 10, 25, 20),
    ),
    "op_mul_rank4_ones_mul_negative": lambda: (
        torch.ones(1, 10, 25, 20),
        (-1) * torch.ones(1, 10, 25, 20),
    ),
    "op_mul_rank4_negative_large_rand": lambda: (
        (-200) * torch.rand(1, 10, 25, 20),
        torch.rand(1, 1, 1, 20),
    ),
    "op_mul_rank4_large_randn": lambda: (
        200 * torch.randn(1, 10, 25, 20),
        torch.rand(1, 10, 25, 1),
    ),
    "op_mul_rank4_randn_mutltiple_broadcasts": lambda: (
        torch.randn(1, 4, 4, 1),
        torch.randn(1, 1, 4, 4),
    ),
}


test_data_suite_2 = {
    # (test_name, input, other,) See torch.mul() for info
    "op_mul_rank2_rand": lambda: (
        torch.rand(4, 5),
        torch.rand(5),
    ),
    "op_mul_rank3_randn": lambda: (
        torch.randn(10, 5, 2),
        torch.randn(5, 2),
    ),
    "op_mul_rank4_randn": lambda: (
        torch.randn(1, 10, 25, 20),
        torch.randn(1, 25, 20),
    ),
    "op_mul_rank4_randn_2": lambda: (
        torch.randn(1, 25, 1),
        torch.randn(1, 3, 25, 10),
    ),
}


test_data_suite_int32 = {
    # (test_name, input, other,) See torch.mul() for info
    "op_mul_rank4_randn_int32": lambda: (
        torch.randint(0, 10, (1, 10, 25, 20), dtype=torch.int32),
        torch.randint(0, 10, (1, 10, 25, 20), dtype=torch.int32),
    ),
    "op_mul_rank4_randn_mutltiple_broadcasts_int32": lambda: (
        torch.randint(0, 10, (1, 4, 4, 1), dtype=torch.int32),
        torch.randint(0, 10, (1, 1, 4, 4), dtype=torch.int32),
    ),
    "op_mul_rank4_randn_broadcast_int32": lambda: (
        torch.randint(0, 10, (1, 10, 25, 20), dtype=torch.int32),
        torch.randint(0, 10, (1, 25, 20), dtype=torch.int32),
    ),
}


class Mul(torch.nn.Module):

    def forward(
        self,
        input_: torch.Tensor,
        other_: torch.Tensor,
    ):
        return input_ * other_


@common.parametrize("test_data", test_data_suite)
def test_mul_tensor_tosa_FP(test_data: torch.Tensor):
    pipeline = TosaPipelineFP[input_t1](
        Mul(),
        test_data(),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite_2)
def test_mul_tensor_tosa_FP_diff_input_ranks(test_data: torch.Tensor):
    pipeline = TosaPipelineFP[input_t1](
        Mul(),
        test_data(),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


# MLETORCH-1274 Improve data type checks during partitioning
# view/RESHAPE of integer tensor is not supported for +FP profile which causes issues
# with view_copy (RESHAPE) which isn't supported in FP so removing the int32 tests
# to allow for the dtype validation patches to land.
# filter out the 'op_mul_rank4_randn_int32' only
test_data_int32_without_broadcasting = {
    k: v for k, v in test_data_suite_int32.items() if k != "op_mul_rank4_randn_int32"
}


@common.parametrize("test_data", test_data_int32_without_broadcasting)
def test_mul_tensor_tosa_FP_int32(test_data: torch.Tensor):
    pipeline = TosaPipelineFP[input_t1](
        Mul(),
        test_data(),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite_2)
def test_mul_tensor_tosa_INT_diff_input_ranks(test_data: torch.Tensor):
    pipeline = TosaPipelineINT[input_t1](
        Mul(),
        test_data(),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_mul_tensor_tosa_INT(test_data: torch.Tensor):
    pipeline = TosaPipelineINT[input_t1](
        Mul(),
        test_data(),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite_int32)
def test_mul_tensor_tosa_INT_int32(test_data: torch.Tensor):
    pipeline = TosaPipelineINT[input_t1](
        Mul(),
        test_data(),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_mul_tensor_u55_INT(test_data: torch.Tensor):
    pipeline = EthosU55PipelineINT[input_t1](
        Mul(),
        test_data(),
        aten_op,
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_mul_tensor_u85_INT(test_data: torch.Tensor):
    pipeline = EthosU85PipelineINT[input_t1](
        Mul(),
        test_data(),
        aten_op,
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite_int32)
@common.XfailIfNoCorstone300
def test_mul_tensor_u55_INT_int32(test_data: torch.Tensor):
    pipeline = EthosU55PipelineINT[input_t1](
        Mul(),
        test_data(),
        aten_op,
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite_int32)
@common.XfailIfNoCorstone320
def test_mul_tensor_u85_INT_int32(test_data: torch.Tensor):
    pipeline = EthosU85PipelineINT[input_t1](
        Mul(),
        test_data(),
        aten_op,
        exir_ops=[],
    )
    pipeline.run()


# view/RESHAPE of integer tensor is not supported for +FP profile which causes issues
# with view_copy (RESHAPE) which isn't supported in FP so removing the int32 tests
# to allow for the dtype validation patches to land.


@common.parametrize(
    "test_data",
    test_data_suite | test_data_suite_2 | test_data_int32_without_broadcasting,
)
@common.SkipIfNoModelConverter
def test_mul_tensor_vgf_no_quant(test_data: torch.Tensor):
    pipeline = VgfPipeline[input_t1](
        Mul(),
        test_data(),
        aten_op,
        exir_op=[],
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite | test_data_suite_2)
@common.SkipIfNoModelConverter
def test_mul_tensor_vgf_quant(test_data: torch.Tensor):
    pipeline = VgfPipeline[input_t1](
        Mul(),
        test_data(),
        aten_op,
        exir_op=[],
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite_int32)
@common.SkipIfNoModelConverter
def test_mul_tensor_vgf_quant_int32(test_data: torch.Tensor):
    pipeline = VgfPipeline[input_t1](
        Mul(),
        test_data(),
        aten_op,
        exir_op=[],
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_mul_tensor_16a8w_tosa_INT(test_data: input_t1):
    """Test mul operation with 16A8W quantization (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = TosaPipelineINT[input_t1](
        Mul(),
        test_data(),
        aten_op,
        exir_op=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
        tosa_extensions=["int16"],
    )

    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_mul_tensor_16a8w_u55_INT(test_data: input_t1):
    """Test mul operation with 16A8W quantization on U55 (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = EthosU55PipelineINT[input_t1](
        Mul(),
        test_data(),
        aten_op,
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
        a16w8_quantization=True,
    )

    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_mul_tensor_16a8w_u85_INT(test_data: input_t1):
    """Test mul operation with 16A8W quantization on U85 (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = EthosU85PipelineINT[input_t1](
        Mul(),
        test_data(),
        aten_op,
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
        a16w8_quantization=True,
    )
    pipeline.run()
