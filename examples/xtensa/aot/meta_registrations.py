# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
from executorch.exir.scalar_type import ScalarType
from torch.library import impl, Library

from .utils import get_conv1d_output_size

lib = Library("xtensa", "DEF")

lib.define(
    "quantize_per_tensor(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype) -> (Tensor Z)"
)
lib.define(
    "quantize_per_tensor.out(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype, *, Tensor(a!) out) -> Tensor(a!)"
)

lib.define(
    "dequantize_per_tensor(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype) -> (Tensor Z)"
)
lib.define(
    "dequantize_per_tensor.out(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype, *, Tensor(a!) out) -> Tensor(a!)"
)

lib.define(
    "quantized_layer_norm(Tensor X, Tensor X_scale, Tensor X_zero_point, int[] normalized_shape, Tensor weight, Tensor bias, float eps, float output_scale, int output_zero_point) -> (Tensor Y)"
)

lib.define(
    "quantized_layer_norm.out(Tensor X, Tensor X_scale, Tensor X_zero_point, int[] normalized_shape, Tensor weight, Tensor bias, float eps, float output_scale, int output_zero_point, *, Tensor(a!) out) -> Tensor (a!)"
)

lib.define(
    "quantized_linear(Tensor src, Tensor weight, Tensor bias, int src_zero_point, Tensor weight_zero_point, Tensor out_multiplier, Tensor out_shift, int out_zero_point, Tensor? offset) -> (Tensor Z)"
)
lib.define(
    "quantized_linear.out(Tensor src, Tensor weight, Tensor bias, int src_zero_point, Tensor weight_zero_point, Tensor out_multiplier, Tensor out_shift, int out_zero_point, Tensor? offset, *, Tensor(a!) out) ->  Tensor(a!)"
)

lib.define(
    "quantized_relu(Tensor X, Tensor X_zero_point) -> (Tensor Y)"
)

lib.define(
    "quantized_relu.out(Tensor X, Tensor X_zero_point, *, Tensor(a!) out) -> Tensor (a!)"
)

lib.define(
    "quantized_conv(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, Tensor weight_zero_point, Tensor bias_scale, float out_scale, int out_zero_point, Tensor out_multiplier, Tensor out_shift, bool channel_last=False) -> (Tensor Z)"
)
lib.define(
    "quantized_conv.out(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, Tensor weight_zero_point, Tensor bias_scale, float out_scale, int out_zero_point, Tensor out_multiplier, Tensor out_shift, bool channel_last=False, *, Tensor(a!) out) -> Tensor(a!)"
)

m = Library("xtensa", "IMPL", "Meta")


@impl(m, "quantize_per_tensor")
def quantize_per_tensor_meta(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: ScalarType,
):
    return input.new_empty(input.size(), dtype=dtype)


@impl(m, "dequantize_per_tensor")
def dequantize_per_tensor_meta(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: ScalarType,
):
    return input.new_empty(input.size(), dtype=torch.float)


@impl(m, "quantized_linear")
def quantized_linear_meta(
    src: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    in_zero_point: int,
    weight_zero_point: torch.Tensor,
    out_multiplier: torch.Tensor,
    out_shift: torch.Tensor,
    out_zero_point: int,
    offset: Optional[torch.Tensor],
):
    # src comes in shape [leading_dims, in_dim]
    # weight comes in shape [out_dim, in_dim]
    # output comes in empty with shape [leading_dims, out_dim]
    out_size = list(src.size())
    weight_size = list(weight.size())
    assert len(weight_size) == 2
    out_size[-1] = weight_size[0]
    return src.new_empty(out_size, dtype=torch.uint8)


@impl(m, "quantized_conv")
def quantized_conv_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: Tuple[int],
    padding: Tuple[int],
    dilation: Tuple[int],
    groups: int,
    in_zero_point: int,
    weight_zero_point: torch.Tensor,
    bias_scale: torch.Tensor,
    output_scale: float,
    output_zero_point: int,
    out_multiplier: torch.Tensor,
    out_shift: torch.Tensor,
    channel_last: bool = False,
):
    out_channels, _in_channels, *kernel_size = weight.shape
    in_size = input.shape
    # Assert that the input tensor has at least 3 dimensions, and at most 6
    assert len(in_size) > 2
    assert len(in_size) < 6

    # Compute the output tensor size
    output_size = get_conv1d_output_size(
        in_size, out_channels, stride[0], padding[0], dilation[0], kernel_size[0]
    )

    return input.new_empty(output_size, dtype=input.dtype)


@impl(m, "quantized_layer_norm")
def quantized_layer_norm_meta(
    input: torch.Tensor,
    X_scale: torch.Tensor,
    X_zero_point: torch.Tensor,
    normalized_shape: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    output_scale: float,
    output_zero_point: int,
):
    return input.new_empty(input.size(), dtype=torch.uint8)


@impl(m, "quantized_relu")
def quantized_relu_meta(
    X: torch.Tensor,
    X_zero_point: torch.Tensor,
):
    return X.new_empty(X.size(), dtype=torch.uint8)
