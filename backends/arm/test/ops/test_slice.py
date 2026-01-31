# Copyright 2024-2026 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch

from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_a16w8_quantization_config,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op = "torch.ops.aten.slice.Tensor"
exir_op = "executorch_exir_dialects_edge__ops_aten_slice_copy"

input_t1 = Tuple[torch.Tensor]  # Input x

test_data_suite = {
    "ones_neg_3": lambda: (torch.ones(10), [(3, -3)]),
    "ones_neg_8": lambda: (torch.ones(10), [(-8, 3)]),
    "ones_slice_2": lambda: (torch.ones(10, 10), [(1, 3), (3, 10)]),
    "ones_slice_3": lambda: (torch.ones(10, 10, 10), [(0, 7), (0, 10), (0, 8)]),
    "ones_slice_4": lambda: (
        torch.ones((1, 12, 10, 10)),
        [(0, 1), (0, 5), (3, 5), (4, 10)],
    ),
}


class Slice(torch.nn.Module):
    def forward(self, x: torch.Tensor, s: list[tuple[int, int]]):
        slices = [slice(*i) for i in s]
        return x[slices]


@common.parametrize("test_data", test_data_suite)
def test_slice_tensor_tosa_FP(test_data: torch.Tensor):
    pipeline = TosaPipelineFP[input_t1](Slice(), test_data(), aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_slice_tensor_tosa_INT_nchw(test_data: torch.Tensor):
    pipeline = TosaPipelineINT[input_t1](
        Slice(),
        test_data(),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_slice_tensor_tosa_INT_nhwc(test_data: torch.Tensor):
    pipeline = TosaPipelineINT[input_t1](
        Slice(),
        test_data(),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_slice_tensor_u55_INT(test_data: torch.Tensor):
    pipeline = EthosU55PipelineINT[input_t1](
        Slice(),
        test_data(),
        aten_ops=[],
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_slice_tensor_u85_INT(test_data: torch.Tensor):
    pipeline = EthosU85PipelineINT[input_t1](
        Slice(),
        test_data(),
        aten_ops=[],
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_slice_tensor_vgf_no_quant(test_data: torch.Tensor):
    pipeline = VgfPipeline[input_t1](
        Slice(),
        test_data(),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_slice_tensor_vgf_quant(test_data: torch.Tensor):
    pipeline = VgfPipeline[input_t1](
        Slice(),
        test_data(),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_slice_tensor_16a8w_tosa_INT(test_data: torch.Tensor):
    """Test slice operation with 16A8W quantization (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = TosaPipelineINT[input_t1](
        Slice(),
        test_data(),
        aten_op,
        exir_op=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
        tosa_extensions=["int16"],
    )
    pipeline.quantizer.set_global(
        get_symmetric_a16w8_quantization_config(is_per_channel=per_channel_quantization)
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_slice_tensor_16a8w_u55_INT(test_data: torch.Tensor):
    """Test slice operation with 16A8W quantization on U55 (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = EthosU55PipelineINT[input_t1](
        Slice(),
        test_data(),
        aten_ops=[],
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.quantizer.set_global(
        get_symmetric_a16w8_quantization_config(is_per_channel=per_channel_quantization)
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_slice_tensor_16a8w_u85_INT(test_data: torch.Tensor):
    """Test slice operation with 16A8W quantization on U85 (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = EthosU85PipelineINT[input_t1](
        Slice(),
        test_data(),
        aten_ops=[],
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.quantizer.set_global(
        get_symmetric_a16w8_quantization_config(is_per_channel=per_channel_quantization)
    )
    pipeline.run()
