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

test_data_suite = {
    # (test_name, test_data, [kernel_size, stride, padding])
    "zeros": lambda: (torch.zeros(1, 1, 4, 8), [(4, 6), 2, (2, 0)]),
    "ones": lambda: (torch.ones(1, 16, 50, 32), [4, 2, 0]),
    "rand": lambda: (torch.rand(1, 16, 52, 16), [4, 3, 0]),
    "non_divisible": lambda: (torch.rand(1, 16, 112, 112), [3, 2, 1]),
    "non_divisible_window_height": lambda: (torch.rand(1, 16, 56, 56), [3, (2, 1), 1]),
    "non_divisible_window_width": lambda: (torch.rand(1, 16, 56, 56), [3, (1, 2), 1]),
    "non_divisible_ceil_mode": lambda: (
        torch.rand(1, 16, 112, 112),
        [3, 2, 1, 1, True],
    ),
    "non_divisible_window_height_ceil_mode": lambda: (
        torch.rand(1, 16, 56, 56),
        [3, (2, 1), 1, 1, True],
    ),
    "non_divisible_window_width_ceil_mode": lambda: (
        torch.rand(1, 16, 56, 56),
        [3, (1, 2), 1, 1, True],
    ),
    "non_divisible_window_adjust_padding": lambda: (
        torch.rand(1, 16, 112, 112),
        [3, 2, 1],
    ),
    "non_divisible_window_height_adjust_padding": lambda: (
        torch.rand(1, 16, 56, 56),
        [3, (2, 1), 1],
    ),
    "non_divisible_window_width_adjust_padding": lambda: (
        torch.rand(1, 16, 56, 56),
        [3, (1, 2), 1],
    ),
    "non_divisble_no_padding": lambda: (torch.rand(1, 16, 56, 56), [3, 2, 0]),
    "non_divisible_window_adjust_padding+input": lambda: (
        torch.rand(1, 16, 54, 54),
        [3, 3, 1],
    ),
    "non_divisible_window_height_adjust_padding+input": lambda: (
        torch.rand(1, 16, 54, 54),
        [3, (3, 1), 1],
    ),
    "non_divisible_window_width_adjust_padding+input": lambda: (
        torch.rand(1, 16, 54, 54),
        [3, (1, 3), 1],
    ),
    "randn": lambda: (torch.randn(5, 16, 50, 32), [4, 2, 0]),
}

test_data_suite_bf16 = {
    "rand_bf16": lambda: (
        torch.rand(1, 8, 20, 20, dtype=torch.bfloat16),
        [3, 2, 1],
    ),
}


test_data_suite_dilation = [
    # Simple dilation=2 on 8x8 input, kernel=3, stride=1, no padding
    ("dilation2", torch.rand(1, 1, 8, 8), [3, 1, 0, 2]),
    # Input is 6x6, kernel=3, stride=1, dilation=2.
    # Padding=1 expands the effective input to 8x8.
    ("pad_then_dil2", torch.rand(1, 1, 6, 6), [3, 1, 1, 2]),
    # Input is 16x16, kernel=2x2, stride=2x2, dilation=1 (no dilation).
    # Padding of 1 ensures the input size remains divisible by stride
    # after padding.
    ("even_kernel_fast", torch.rand(1, 3, 16, 16), [(2, 2), (2, 2), (1, 1), 1]),
    # Multi-batch, multi-channel input (N=4, C=3), kernel=3x3,
    # stride=3x3, no padding, dilation=1.
    ("mb_ch_dil1", torch.rand(4, 3, 12, 12), [(3, 3), (3, 3), 0, 1]),
]

aten_op = "torch.ops.aten.max_pool2d.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_max_pool2d_default"

input_t1 = Tuple[torch.Tensor]


class MaxPool2d(torch.nn.Module):
    def __init__(
        self,
        kernel_size: int | Tuple[int, int],
        stride: int | Tuple[int, int],
        padding: int | Tuple[int, int],
        dilation: int | Tuple[int, int] = 1,
        ceil_mode: bool = False,
    ):
        super().__init__()
        self.max_pool_2d = torch.nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )

    def forward(self, x):
        return self.max_pool_2d(x)


@common.parametrize("test_data", test_data_suite | test_data_suite_bf16)
def test_max_pool2d_tosa_FP(test_data: torch.Tensor):
    test_data, model_params = test_data()
    pipeline = TosaPipelineFP[input_t1](
        MaxPool2d(*model_params),
        (test_data,),
        aten_op,
        exir_op,
        tosa_extensions=["bf16"],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_max_pool2d_tosa_INT(test_data: torch.Tensor):
    test_data, model_params = test_data()
    pipeline = TosaPipelineINT[input_t1](
        MaxPool2d(*model_params),
        (test_data,),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_max_pool2d_tosa_INT_a16w8(test_data: torch.Tensor):
    """Test max_pool2d operation with int16 I/O quantization for TOSA INT."""
    test_data, model_params = test_data()
    pipeline = TosaPipelineINT[input_t1](
        MaxPool2d(*model_params),
        (test_data,),
        aten_op,
        exir_op,
        tosa_extensions=["int16"],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_max_pool2d_u55_INT(test_data: torch.Tensor):
    test_data, model_params = test_data()
    EthosU55PipelineINT[input_t1](
        MaxPool2d(*model_params),
        (test_data,),
        aten_op,
        exir_ops=[],
    ).run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_max_pool2d_16a8w_u55_INT(test_data: torch.Tensor):
    """Test max_pool2d with 16A8W quantization on U55 (16-bit activations, 8-bit weights)"""
    test_data, model_params = test_data()
    pipeline = EthosU55PipelineINT[input_t1](
        MaxPool2d(*model_params),
        (test_data,),
        aten_op,
        exir_ops=[],
        per_channel_quantization=False,
        a16w8_quantization=True,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_max_pool2d_u85_INT(test_data: torch.Tensor):
    test_data, model_params = test_data()
    EthosU85PipelineINT[input_t1](
        MaxPool2d(*model_params),
        (test_data,),
        aten_op,
        exir_ops=[],
    ).run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_max_pool2d_16a8w_u85_INT(test_data: torch.Tensor):
    """Test max_pool2d with 16A8W quantization on U85 (16-bit activations, 8-bit weights)"""
    test_data, model_params = test_data()
    pipeline = EthosU85PipelineINT[input_t1](
        MaxPool2d(*model_params),
        (test_data,),
        aten_op,
        exir_ops=[],
        per_channel_quantization=False,
        a16w8_quantization=True,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


reject_data_suite = {
    "reject_1": lambda: (MaxPool2d(1, 4, 0), torch.rand(1, 10, 10, 10)),
    "reject_2": lambda: (MaxPool2d((1, 257), 1, 0), torch.rand(1, 16, 5, 300)),
    "reject_3": lambda: (MaxPool2d((800, 90), 1, 0), torch.rand(1, 16, 850, 100)),
}


@common.parametrize("test_data", reject_data_suite)
@common.XfailIfNoCorstone300
def test_max_pool2d_u55_INT_failure_set(test_data: Tuple):
    module, test_data = test_data()
    pipeline = EthosU55PipelineINT[input_t1](
        module,
        (test_data,),
        aten_op,
        exir_op,
        run_on_fvp=False,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.pop_stage("check_count.exir")
    pipeline.run()


# Convert the list of (name, tensor, params) into the dict-of-lambdas shape
dilation_test_data = {
    name: (lambda data=data, params=params: (data, params))
    for name, data, params in test_data_suite_dilation
}


@common.parametrize("test_data", dilation_test_data)
def test_max_pool2d_tosa_FP_dilation(test_data):
    """
    TOSA FP pipeline with dilation > 1 (and dilation=1 sanity cases).
    """
    data, model_params = test_data()
    pipeline = TosaPipelineFP[input_t1](
        MaxPool2d(*model_params),
        (data,),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", dilation_test_data)
def test_max_pool2d_tosa_INT_dilation(test_data):
    """
    TOSA INT pipeline with dilation > 1 (and dilation=1 sanity cases).
    """
    data, model_params = test_data()
    pipeline = TosaPipelineINT[input_t1](
        MaxPool2d(*model_params),
        (data,),
        aten_op,
        exir_op,
        symmetric_io_quantization=True,
    )
    pipeline.run()


# VGF tests
@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_max_pool2d_vgf_no_quant(test_data: torch.Tensor):
    test_data, model_params = test_data()
    pipeline = VgfPipeline[input_t1](
        MaxPool2d(*model_params),
        (test_data,),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_max_pool2d_vgf_quant(test_data: torch.Tensor):
    test_data, model_params = test_data()
    pipeline = VgfPipeline[input_t1](
        MaxPool2d(*model_params),
        (test_data,),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", dilation_test_data)
@common.SkipIfNoModelConverter
def test_max_pool2d_vgf_no_quant_dilation(test_data: torch.Tensor):
    """
    VGF FP pipeline with dilation > 1 (and dilation=1 sanity cases).
    """
    test_data, model_params = test_data()
    pipeline = VgfPipeline[input_t1](
        MaxPool2d(*model_params),
        (test_data,),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", dilation_test_data)
@common.SkipIfNoModelConverter
def test_max_pool2d_vgf_quant_dilation(test_data: torch.Tensor):
    """
    VGF INT pipeline with dilation > 1 (and dilation=1 sanity cases).
    """
    test_data, model_params = test_data()
    pipeline = VgfPipeline[input_t1](
        MaxPool2d(*model_params),
        (test_data,),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()
