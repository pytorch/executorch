# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from math import prod
from typing import Sequence

import torch
import torch.nn.functional as F
from executorch.backends.cortex_m.passes.passes_utils import (
    is_channel_broadcast,
    requantize_cmsis,
    SHIFT_INT8,
)
from executorch.exir.dialects._ops import ops as exir_ops

# To provide the implementation of the operators
from torch.library import impl, Library, register_fake


# New operator library with a custom namespace to allow fusion etc.
lib = Library("cortex_m", "DEF")

###
# dequantize_per_tensor
###

lib.define(
    "quantize_per_tensor(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype) -> (Tensor Z)"
)

lib.define(
    "quantize_per_tensor.out(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype, *, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("cortex_m::quantize_per_tensor")
def quantize_per_tensor_meta(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty_like(input, dtype=dtype)


@impl(lib, "quantize_per_tensor", "CompositeExplicitAutograd")
def quantize_per_tensor_impl(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    The implementation of the quantize_per_tensor operator is the same as the
    quantize_per_tensor operator in the edge dialect.
    """
    return exir_ops.edge.quantized_decomposed.quantize_per_tensor.default(
        input, scale, zero_point, quant_min, quant_max, dtype
    )


###
# dequantize_per_tensor
###

lib.define(
    "dequantize_per_tensor(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype) -> (Tensor Z)"
)
lib.define(
    "dequantize_per_tensor.out(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype, *, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("cortex_m::dequantize_per_tensor")
def dequantize_per_tensor_meta(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty_like(input, dtype=torch.float)


@impl(lib, "dequantize_per_tensor", "CompositeExplicitAutograd")
def dequantize_per_tensor_impl(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    The implementation of the dequantize_per_tensor operator is the same as the
    dequantize_per_tensor operator in the edge dialect.
    """
    return exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default(
        input, scale, zero_point, quant_min, quant_max, dtype
    )


# Define the operator schema with multipliers and shifts (11 args)
lib.define(
    "quantized_add("
    "Tensor self, Scalar self_zero_point, Scalar self_multiplier, Scalar self_shift, "
    "Tensor other, Scalar other_zero_point, Scalar other_multiplier, Scalar other_shift, "
    "Scalar output_zero_point, Scalar output_multiplier, Scalar output_shift) -> Tensor"
)

# Define the operator schema with multipliers and shifts (11 args + out tensor)
lib.define(
    "quantized_add.out("
    "Tensor self, Scalar self_zero_point, Scalar self_multiplier, Scalar self_shift, "
    "Tensor other, Scalar other_zero_point, Scalar other_multiplier, Scalar other_shift, "
    "Scalar output_zero_point, Scalar output_multiplier, Scalar output_shift, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("cortex_m::quantized_add")
def quantized_add_meta(
    self: torch.Tensor,
    self_zero_point: int,
    self_multiplier: int,
    self_shift: int,
    other: torch.Tensor,
    other_zero_point: int,
    other_multiplier: int,
    other_shift: int,
    output_zero_point: int,
    output_multiplier: int,
    output_shift: int,
) -> torch.Tensor:
    assert self.shape == other.shape or is_channel_broadcast(self, other), (
        "Cortex-M quantized_add: broadcasting is not yet supported except for channel dim — "
        f"got self.shape={self.shape}, other.shape={other.shape}"
    )
    if self.numel() > other.numel():
        output_tensor = self
    else:
        output_tensor = other
    return torch.empty_like(output_tensor)


@impl(lib, "quantized_add", "CompositeExplicitAutograd")
def quantized_add_impl(
    self: torch.Tensor,
    self_zero_point: int,
    self_multiplier: int,
    self_shift: int,
    other: torch.Tensor,
    other_zero_point: int,
    other_multiplier: int,
    other_shift: int,
    output_zero_point: int,
    output_multiplier: int,
    output_shift: int,
) -> torch.Tensor:
    assert self.shape == other.shape or is_channel_broadcast(self, other), (
        "Cortex-M quantized_add: broadcasting is not yet supported except for channel dim — "
        f"got self.shape={self.shape}, other.shape={other.shape}"
    )
    self_shifted = (self.to(torch.int32) - self_zero_point) << SHIFT_INT8
    self_fp = requantize_cmsis(self_shifted, self_multiplier, self_shift)

    other_shifted = (other.to(torch.int32) - other_zero_point) << SHIFT_INT8
    other_fp = requantize_cmsis(other_shifted, other_multiplier, other_shift)

    result_fp = self_fp + other_fp
    result_quantized = requantize_cmsis(result_fp, output_multiplier, output_shift)
    result = torch.clamp(result_quantized + output_zero_point, -128, 127).to(torch.int8)
    return result


# ===================================================================
# QUANTIZED MUL OPERATION DEFINITION
# ===================================================================
lib.define(
    "quantized_mul("
    "Tensor self, Scalar self_zero_point, "
    "Tensor other, Scalar other_zero_point, "
    "Scalar output_zero_point, Scalar output_multiplier, Scalar output_shift) -> Tensor"
)
lib.define(
    "quantized_mul.out("
    "Tensor self, Scalar self_zero_point, "
    "Tensor other, Scalar other_zero_point, "
    "Scalar output_zero_point, Scalar output_multiplier, Scalar output_shift, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("cortex_m::quantized_mul")
def quantized_mul_meta(
    self: torch.Tensor,
    self_zero_point: int,
    other: torch.Tensor,
    other_zero_point: int,
    output_zero_point: int,
    output_multiplier: int,
    output_shift: int,
) -> torch.Tensor:
    # Broadcast to output shape
    assert self.shape == other.shape or is_channel_broadcast(self, other), (
        "Cortex-M quantized_mul: broadcasting is not yet supported except for channel dim — "
        f"got self.shape={self.shape}, other.shape={other.shape}"
    )
    if self.numel() > other.numel():
        output_tensor = self
    else:
        output_tensor = other
    return torch.empty_like(output_tensor)


@impl(lib, "quantized_mul", "CompositeExplicitAutograd")
def quantized_mul_impl(
    self: torch.Tensor,
    self_zero_point: int,
    other: torch.Tensor,
    other_zero_point: int,
    output_zero_point: int,
    output_multiplier: int,
    output_shift: int,
) -> torch.Tensor:
    # CMSIS-NN kernel multiplies raw int8 tensors (after zero-point offset) and
    # only uses the output multiplier/shift for rescaling. Mirror that here to
    # keep the composite implementation numerically aligned with the backend.
    assert self.shape == other.shape or is_channel_broadcast(self, other), (
        "Cortex-M quantized_mul: broadcasting is not yet supported except for channel dim — "
        f"got self.shape={self.shape}, other.shape={other.shape}"
    )
    self_int = self.to(torch.int32) - self_zero_point
    other_int = other.to(torch.int32) - other_zero_point
    result_fp = self_int * other_int
    result_quantized = requantize_cmsis(result_fp, output_multiplier, output_shift)
    result = torch.clamp(result_quantized + output_zero_point, -128, 127).to(torch.int8)
    return result


# ===================================================================
# MINIMUM/MAXIMUM OPERATION DEFINITIONS
# ===================================================================
lib.define("minimum(Tensor self, Tensor other) -> Tensor")
lib.define("minimum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("cortex_m::minimum")
def minimum_meta(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    assert self.dtype == other.dtype, (
        "Cortex-M minimum: dtype mismatch — "
        f"got self.dtype={self.dtype}, other.dtype={other.dtype}"
    )
    broadcasted_shape = torch.broadcast_shapes(self.shape, other.shape)
    return torch.empty(broadcasted_shape, dtype=self.dtype, device=self.device)


@impl(lib, "minimum", "CompositeExplicitAutograd")
def minimum_impl(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return torch.minimum(self, other)


lib.define("maximum(Tensor self, Tensor other) -> Tensor")
lib.define("maximum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("cortex_m::maximum")
def maximum_meta(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    assert self.dtype == other.dtype, (
        "Cortex-M maximum: dtype mismatch — "
        f"got self.dtype={self.dtype}, other.dtype={other.dtype}"
    )
    broadcasted_shape = torch.broadcast_shapes(self.shape, other.shape)
    return torch.empty(broadcasted_shape, dtype=self.dtype, device=self.device)


@impl(lib, "maximum", "CompositeExplicitAutograd")
def maximum_impl(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return torch.maximum(self, other)


# ===================================================================
# QUANTIZED LINEAR OPERATION DEFINITION
# ===================================================================

lib.define(
    "quantized_linear.out("
    "Tensor input,  "
    "Tensor weights, "
    "Tensor? bias, "
    "Tensor? kernel_sum, "
    "Scalar input_offset, "
    "Scalar filter_offset, "
    "Scalar output_offset, "
    "int[] requantize_multipliers, "
    "int[] requantize_shifts, "
    "Scalar activation_max, "
    "Scalar activation_min, "
    "*, Tensor(a!) out"
    ") -> Tensor(a!)"
)

# Define functional variant (non-out version)
lib.define(
    "quantized_linear("
    "Tensor input,  "
    "Tensor weights, "
    "Tensor? bias, "
    "Tensor? kernel_sum, "
    "Scalar input_offset, "
    "Scalar filter_offset, "
    "Scalar output_offset, "
    "int[] requantize_multipliers, "
    "int[] requantize_shifts, "
    "Scalar activation_max, "
    "Scalar activation_min"
    ") -> Tensor"
)


# Fake meta function for shape inference (functional variant)
@register_fake("cortex_m::quantized_linear")
def quantized_linear_meta(
    input,
    weights,
    bias,
    kernel_sum,
    input_offset,
    filter_offset,
    output_offset,
    requantize_multipliers,
    requantize_shifts,
    activation_max,
    activation_min,
) -> torch.Tensor:

    shape = (*input.shape[:-1], weights.shape[0])
    return torch.empty(shape, dtype=input.dtype, device=input.device)


# Functional variant implementation
@impl(lib, "quantized_linear", "CompositeExplicitAutograd")
def quantized_linear_impl(
    input: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    kernel_sum: torch.Tensor,
    input_offset: int,
    filter_offset: int,
    output_offset: int,
    requantize_multipliers: torch.Tensor,
    requantize_shifts: torch.Tensor,
    activation_max: int,
    activation_min: int,
) -> torch.Tensor:
    """
    Functional variant - creates output tensor and calls out variant
    """

    # Leaving both implementations for debugging purposes.
    compute_using_kernel_sum = True

    if compute_using_kernel_sum:
        weights_int32 = weights.to(torch.int32)

        input_int32 = input.to(torch.int32)
        new_shape = (prod(input.shape[:-1]), input.shape[-1])
        input_reshaped = input_int32.reshape(new_shape)

        lhs_sum = torch.sum(input_reshaped, dim=-1, keepdim=True) * filter_offset
        output = torch.mm(input_reshaped, weights_int32.T) + lhs_sum + kernel_sum
        output_shape = (*input.shape[:-1], output.shape[-1])
        output_reshaped = output.reshape(output_shape)
    else:
        weights_int32 = weights.to(torch.int32) + filter_offset

        input_int32 = input.to(torch.int32) + input_offset
        new_shape = (prod(input.shape[:-1]), input.shape[-1])
        input_reshaped = input_int32.reshape(new_shape)

        output = torch.mm(input_reshaped, weights_int32.T)
        if bias is not None:
            output = output + bias
        output_shape = (*input.shape[:-1], output.shape[-1])
        output_reshaped = output.reshape(output_shape)

    output = requantize_cmsis(
        output_reshaped, requantize_multipliers[0], requantize_shifts[0]
    )
    output += output_offset
    output = torch.clamp(output, activation_min, activation_max).to(torch.int8)
    return output


# ===================================================================
# TRANSPOSE OPERATION DEFINITION
# ===================================================================
lib.define("transpose(Tensor input, int[] perm) -> Tensor")
lib.define("transpose.out(Tensor input, int[] perm, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("cortex_m::transpose")
def transpose_meta(input: torch.Tensor, perm) -> torch.Tensor:
    output_shape = [input.shape[idx] for idx in perm]
    return torch.empty(output_shape, dtype=input.dtype, device=input.device)


@impl(lib, "transpose", "CompositeExplicitAutograd")
def transpose_impl(input: torch.Tensor, perm) -> torch.Tensor:
    return input.permute(tuple(perm)).contiguous()


# ===================================================================
# QUANTIZED CONV2D OPERATION DEFINITION
# ===================================================================

lib.define(
    "quantized_conv2d("
    "Tensor input, "
    "Tensor weight, "
    "Tensor? bias, "
    "int[] stride, "
    "int[] padding, "
    "int[] dilation, "
    "int input_offset, "
    "int output_offset, "
    "Tensor requantize_multipliers, "
    "Tensor requantize_shifts, "
    "int activation_min, "
    "int activation_max"
    ") -> Tensor"
)


lib.define(
    "quantized_conv2d.out("
    "Tensor input, "
    "Tensor weight, "
    "Tensor? bias, "
    "int[] stride, "
    "int[] padding, "
    "int[] dilation, "
    "int input_offset, "
    "int output_offset, "
    "Tensor requantize_multipliers, "
    "Tensor requantize_shifts, "
    "int activation_min, "
    "int activation_max, "
    "*, Tensor(a!) out"
    ") -> Tensor(a!)"
)


def _compute_conv2d_output_shape(
    input_shape: torch.Size,
    weight_shape: torch.Size,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
) -> torch.Size:
    batch = input_shape[0]
    in_height = input_shape[2]
    in_width = input_shape[3]
    # We store the weights in OHWI layout (out, kernel_h, kernel_w, in)
    kernel_height = weight_shape[1]
    kernel_width = weight_shape[2]

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation

    out_channels = weight_shape[0]
    out_height = (
        in_height + 2 * pad_h - dilation_h * (kernel_height - 1) - 1
    ) // stride_h + 1
    out_width = (
        in_width + 2 * pad_w - dilation_w * (kernel_width - 1) - 1
    ) // stride_w + 1
    return torch.Size([batch, out_channels, out_height, out_width])


@register_fake("cortex_m::quantized_conv2d")
def quantized_conv2d_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    input_offset: int,
    output_offset: int,
    requantize_multipliers: torch.Tensor,
    requantize_shifts: torch.Tensor,
    activation_min: int,
    activation_max: int,
) -> torch.Tensor:
    stride_vals = list(stride)
    padding_vals = list(padding)
    dilation_vals = list(dilation)
    output_shape = _compute_conv2d_output_shape(
        input.shape, weight.shape, stride_vals, padding_vals, dilation_vals
    )
    return torch.empty(
        output_shape,
        dtype=torch.int8,
        device=input.device,
        memory_format=torch.channels_last,
    )


@impl(lib, "quantized_conv2d", "CompositeExplicitAutograd")
def quantized_conv2d_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    input_offset: int,
    output_offset: int,
    requantize_multipliers: torch.Tensor,
    requantize_shifts: torch.Tensor,
    activation_min: int,
    activation_max: int,
) -> torch.Tensor:
    if input.dim() != 4 or weight.dim() != 4:
        raise RuntimeError("quantized_conv2d expects 4D input and weight tensors")
    # Convert to int32 for accumulation and apply offsets
    input_int32 = input.to(torch.int32) + int(input_offset)
    weight_int32 = weight.to(torch.int32)

    if bias is None:
        bias_int32 = torch.zeros(
            weight.shape[0], dtype=torch.int32, device=input.device
        )
    else:
        bias_int32 = bias.to(torch.int32)

    input_channels = input.shape[1]
    kernel_input_channels = weight.shape[3]
    groups = input_channels // kernel_input_channels

    # Convert weights back to OIHW layout expected by torch.nn.functional.conv2d
    weight_oi_hw = weight_int32.permute(0, 3, 1, 2).contiguous()

    conv_acc = F.conv2d(
        input_int32,
        weight_oi_hw,
        bias_int32,
        stride=tuple(stride),
        padding=tuple(padding),
        dilation=tuple(dilation),
        groups=groups,
    )

    result_channels = []
    for output_channel_i in range(conv_acc.shape[1]):
        result_channel = requantize_cmsis(
            conv_acc[:, output_channel_i, :, :],
            int(requantize_multipliers[output_channel_i]),
            int(requantize_shifts[output_channel_i]),
        )
        result_channels.append(result_channel)

    result = torch.stack(result_channels, dim=1)

    result += output_offset
    result = torch.clamp(result, activation_min, activation_max)

    return result.to(torch.int8)


# ===================================================================
# QUANTIZED DEPTHWISE CONV2D OPERATION DEFINITION
# ===================================================================

lib.define(
    "quantized_depthwise_conv2d("
    "Tensor input, "
    "Tensor weight, "
    "Tensor? bias, "
    "int[] stride, "
    "int[] padding, "
    "int[] dilation, "
    "int groups, "
    "int input_offset, "
    "int output_offset, "
    "Tensor requantize_multipliers, "
    "Tensor requantize_shifts, "
    "int activation_min, "
    "int activation_max"
    ") -> Tensor"
)


lib.define(
    "quantized_depthwise_conv2d.out("
    "Tensor input, "
    "Tensor weight, "
    "Tensor? bias, "
    "int[] stride, "
    "int[] padding, "
    "int[] dilation, "
    "int groups, "
    "int input_offset, "
    "int output_offset, "
    "Tensor requantize_multipliers, "
    "Tensor requantize_shifts, "
    "int activation_min, "
    "int activation_max, "
    "*, Tensor(a!) out"
    ") -> Tensor(a!)"
)


@register_fake("cortex_m::quantized_depthwise_conv2d")
def quantized_depthwise_conv2d_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    input_offset: int,
    output_offset: int,
    requantize_multipliers: torch.Tensor,
    requantize_shifts: torch.Tensor,
    activation_min: int,
    activation_max: int,
) -> torch.Tensor:
    stride_vals = list(stride)
    padding_vals = list(padding)
    dilation_vals = list(dilation)
    output_shape = _compute_conv2d_output_shape(
        input.shape, weight.shape, stride_vals, padding_vals, dilation_vals
    )
    return torch.empty(
        output_shape,
        dtype=torch.int8,
        device=input.device,
        memory_format=torch.channels_last,
    )


@impl(lib, "quantized_depthwise_conv2d", "CompositeExplicitAutograd")
def quantized_depthwise_conv2d_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    input_offset: int,
    output_offset: int,
    requantize_multipliers: torch.Tensor,
    requantize_shifts: torch.Tensor,
    activation_min: int,
    activation_max: int,
) -> torch.Tensor:
    if input.dim() != 4 or weight.dim() != 4:
        raise RuntimeError(
            "quantized_depthwise_conv2d expects 4D input and weight tensors"
        )

    # Validate depthwise convolution constraint: groups == input_channels
    input_channels = input.shape[1]
    if groups != input_channels:
        raise RuntimeError(
            f"quantized_depthwise_conv2d: groups ({groups}) must equal input channels ({input_channels})"
        )

    # Convert to int32 for accumulation and apply offsets
    input_int32 = input.to(torch.int32) + int(input_offset)
    weight_int32 = weight.to(torch.int32)

    if bias is None:
        bias_int32 = torch.zeros(
            weight.shape[0], dtype=torch.int32, device=input.device
        )
    else:
        bias_int32 = bias.to(torch.int32)

    # Convert weights back to OIHW layout expected by torch.nn.functional.conv2d
    weight_oi_hw = weight_int32.permute(0, 3, 1, 2).contiguous()

    # Depthwise convolution has groups == input_channels
    conv_acc = F.conv2d(
        input_int32,
        weight_oi_hw,
        bias_int32,
        stride=tuple(stride),
        padding=tuple(padding),
        dilation=tuple(dilation),
        groups=groups,
    )

    result_channels = []
    for output_channel_i in range(conv_acc.shape[1]):
        result_channel = requantize_cmsis(
            conv_acc[:, output_channel_i, :, :],
            int(requantize_multipliers[output_channel_i]),
            int(requantize_shifts[output_channel_i]),
        )
        result_channels.append(result_channel)

    result = torch.stack(result_channels, dim=1)

    result += output_offset
    result = torch.clamp(result, activation_min, activation_max)

    return result.to(torch.int8)
