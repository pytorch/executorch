# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from math import prod
from typing import Optional, Tuple

import torch
from torch.library import Library, register_fake

from .utils import get_conv1d_output_size, get_conv2d_output_size

lib = Library("cadence", "DEF")

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
    "quantized_relu(Tensor X, Tensor X_zero_point, int out_zero_point, Tensor out_multiplier, Tensor out_shift) -> (Tensor Y)"
)
lib.define(
    "quantized_relu.out(Tensor X, Tensor X_zero_point, int out_zero_point, Tensor out_multiplier, Tensor out_shift, *, Tensor(a!) out) -> Tensor (a!)"
)

lib.define(
    "quantized_conv(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, Tensor weight_zero_point, Tensor bias_scale, float out_scale, int out_zero_point, Tensor out_multiplier, Tensor out_shift, bool channel_last=False) -> (Tensor Z)"
)
lib.define(
    "quantized_conv.out(Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding, int[] dilation, int groups, int input_zero_point, Tensor weight_zero_point, Tensor bias_scale, float out_scale, int out_zero_point, Tensor out_multiplier, Tensor out_shift, bool channel_last=False, *, Tensor(a!) out) -> Tensor(a!)"
)

lib.define(
    "quantized_matmul(Tensor X, int X_zero_point, Tensor Y, int Y_zero_point, Tensor? bias, int out_multiplier, int out_shift, int out_zero_point, bool transposed=False) -> (Tensor Z)"
)
lib.define(
    "quantized_matmul.out(Tensor X, int X_zero_point, Tensor Y, int Y_zero_point, Tensor? bias, int out_multiplier, int out_shift, int out_zero_point, bool transposed=False, *, Tensor(a!) out) -> Tensor(a!)"
)

m = Library("cadence", "IMPL", "Meta")


@register_fake("cadence::quantize_per_tensor")
def quantize_per_tensor_meta(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return input.new_empty(input.size(), dtype=dtype)


@register_fake("cadence::dequantize_per_tensor")
def dequantize_per_tensor_meta(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return input.new_empty(input.size(), dtype=torch.float)


@register_fake("cadence::quantized_linear")
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
) -> torch.Tensor:
    # src comes in shape [leading_dims, in_dim]
    # weight comes in shape [out_dim, in_dim]
    # output comes in empty with shape [leading_dims, out_dim]
    out_size = list(src.size())
    weight_size = list(weight.size())
    assert len(weight_size) == 2
    out_size[-1] = weight_size[0]
    return src.new_empty(out_size, dtype=torch.uint8)


@register_fake("cadence::quantized_conv")
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
) -> torch.Tensor:
    out_channels, _in_channels, *kernel_size = weight.shape
    in_size = input.shape
    # Assert that the input tensor has at least 3 dimensions, and at most 6
    assert len(in_size) > 2
    assert len(in_size) < 6

    # Compute the output tensor size
    output_size = (
        get_conv1d_output_size(
            in_size, out_channels, stride[1], padding[1], dilation[1], kernel_size[0]
        )
        if len(in_size) == 3
        else get_conv2d_output_size(
            in_size, out_channels, stride, padding, dilation, kernel_size, channel_last
        )
    )

    return input.new_empty(output_size, dtype=input.dtype)


@register_fake("cadence::quantized_layer_norm")
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
) -> torch.Tensor:
    return input.new_empty(input.size(), dtype=torch.uint8)


@register_fake("cadence::quantized_relu")
def quantized_relu_meta(
    X: torch.Tensor,
    X_zero_point: torch.Tensor,
    out_zero_point: int,
    out_multiplier: torch.Tensor,
    out_shift: torch.Tensor,
) -> torch.Tensor:
    return X.new_empty(X.size(), dtype=torch.uint8)


@register_fake("cadence::quantized_matmul")
def quantized_matmul_meta(
    X: torch.Tensor,
    X_zero_point: int,
    Y: torch.Tensor,
    Y_zero_point: int,
    bias: Optional[torch.Tensor],
    out_multiplier: int,
    out_shift: int,
    out_zero_point: int,
    transposed: bool = False,
) -> torch.Tensor:
    X_size = list(X.size())
    Y_size = list(Y.size())

    # Get the batch dimensions for both tensors
    X_batch_dims = X_size[:-2]
    Y_batch_dims = Y_size[:-2]

    # If they don't match, check that they're compatible
    if X_batch_dims != Y_batch_dims:
        assert prod(X_batch_dims) == prod(
            Y_batch_dims
        ), f"Batch dimensions of X and Y do not match: {X_batch_dims} vs {Y_batch_dims}"

    # Get the matmul output size
    if transposed:
        assert X_size[-1] == Y_size[-1], "matrices cannot be multiplied"
        mat_size = [X_size[-2], Y_size[-2]]
    else:
        assert X_size[-1] == Y_size[-2], "matrices cannot be multiplied"
        mat_size = [X_size[-2], Y_size[-1]]

    # Combine the larger batch dimensions with the matmul output size
    out_size = (
        X_batch_dims + mat_size
        if len(X_batch_dims) > len(Y_batch_dims)
        else Y_batch_dims + mat_size
    )

    return X.new_empty(out_size, dtype=X.dtype)
