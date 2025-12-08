# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)


class LayerNorm(torch.nn.Module):

    def __init__(
        self,
        normalized_shape: Union[int, List[int]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        has_bias: bool = True,
    ):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=has_bias,
        )
        if elementwise_affine:
            self.layer_norm.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        if has_bias:
            self.layer_norm.bias = torch.nn.Parameter(torch.rand(normalized_shape))

    def forward(self, x):
        return self.layer_norm(x)


input_t = tuple[torch.Tensor]
test_data_suite = {
    "randn_last_dim": lambda: ((torch.randn(1, 5, 5, 5),), LayerNorm([5])),
    "rand_last_two_dims": lambda: ((torch.rand(1, 5, 5, 5),), LayerNorm([5, 5])),
    "rand_last_two_dims_not_elementwise_affine": lambda: (
        (torch.rand(1, 5, 5, 5),),
        LayerNorm([5, 5], 1e-5, False),
    ),
    "rand_last_two_dims_not_elementwise_affine_no_bias": lambda: (
        (torch.rand(1, 5, 5, 5),),
        LayerNorm([5, 5], 1e-5, False, False),
    ),
    "randn_last_three_dims": lambda: (
        (torch.randn(1, 15, 10, 5),),
        LayerNorm([15, 10, 5]),
    ),
    "randn_last_three_dims_no_bias": lambda: (
        (torch.randn(1, 15, 10, 5),),
        LayerNorm([15, 10, 5], 1e-2, False, False),
    ),
}


@common.parametrize("test_data", test_data_suite)
def test_native_layer_norm_tosa_FP(test_data):
    test_data, model = test_data()
    pipeline = TosaPipelineFP[input_t](
        model,
        test_data,
        "torch.ops.aten.layer_norm.default",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_native_layer_norm_tosa_INT(test_data):
    test_data, model = test_data()
    pipeline = TosaPipelineINT[input_t](
        model,
        test_data,
        "torch.ops.aten.sub.Tensor",  # Just check for sub op included in the layernorm decomposition
        symmetric_io_quantization=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_native_layer_norm_u55_INT(test_data):
    test_data, model = test_data()
    pipeline = EthosU55PipelineINT[input_t](
        model,
        test_data,
        "torch.ops.aten.sub.Tensor",  # Just check for sub op included in the layernorm decomposition
        symmetric_io_quantization=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_native_layer_norm_u85_INT(test_data):
    test_data, model = test_data()
    pipeline = EthosU85PipelineINT[input_t](
        model,
        test_data,
        "torch.ops.aten.sub.Tensor",  # Just check for sub op included in the layernorm decomposition
        symmetric_io_quantization=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_native_layer_norm_vgf_FP(test_data):
    test_input, model = test_data()
    pipeline = VgfPipeline[input_t](
        model,
        test_input,
        "torch.ops.aten.layer_norm.default",
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_native_layer_norm_vgf_INT(test_data):
    test_input, model = test_data()
    pipeline = VgfPipeline[input_t](
        model,
        test_input,
        "torch.ops.aten.sub.Tensor",
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_native_layer_norm_tosa_INT_a16w8(test_data):
    """Test layer_norm with int16 I/O quantization for TOSA INT."""
    test_input, model = test_data()
    pipeline = TosaPipelineINT[input_t](
        model,
        test_input,
        "torch.ops.aten.sub.Tensor",  # check for sub op in decomposition
        symmetric_io_quantization=True,
        tosa_extensions=["int16"],
        epsilon=2**16,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_native_layer_norm_16a8w_u55_INT16(test_data):
    """Test layer_norm with int16 I/O quantization for U55"""
    test_input, model = test_data()
    pipeline = EthosU55PipelineINT[input_t](
        model,
        test_input,
        "torch.ops.aten.sub.Tensor",
        symmetric_io_quantization=True,
        a16w8_quantization=True,
        epsilon=2**16,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_native_layer_norm_16a8w_u85_INT16(test_data):
    """Test layer_norm with int16 I/O quantization for U85"""
    test_input, model = test_data()
    pipeline = EthosU85PipelineINT[input_t](
        model,
        test_input,
        "torch.ops.aten.sub.Tensor",
        symmetric_io_quantization=True,
        a16w8_quantization=True,
        epsilon=2**16,
    )
    pipeline.run()
