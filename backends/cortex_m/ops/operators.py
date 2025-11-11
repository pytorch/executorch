# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from math import prod

import torch
from executorch.backends.cortex_m.passes.passes_utils import (
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
    assert self.shape == other.shape, (
        "Cortex-M quantized_mul: broadcasting is not yet supported — "
        f"got self.shape={self.shape}, other.shape={other.shape}"
    )
    broadcasted_shape = torch.broadcast_shapes(self.shape, other.shape)
    return torch.empty(broadcasted_shape, dtype=torch.int8, device=self.device)


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
    assert self.shape == other.shape, (
        "Cortex-M quantized_mul: broadcasting is not yet supported — "
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
    assert self.shape == other.shape, (
        "Cortex-M quantized_mul: broadcasting is not yet supported — "
        f"got self.shape={self.shape}, other.shape={other.shape}"
    )
    broadcasted_shape = torch.broadcast_shapes(self.shape, other.shape)
    return torch.empty(broadcasted_shape, dtype=torch.int8, device=self.device)


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
    assert self.shape == other.shape, (
        "Cortex-M quantized_mul: broadcasting is not yet supported — "
        f"got self.shape={self.shape}, other.shape={other.shape}"
    )
    self_int = self.to(torch.int32) - self_zero_point
    other_int = other.to(torch.int32) - other_zero_point
    result_fp = self_int * other_int
    result_quantized = requantize_cmsis(result_fp, output_multiplier, output_shift)
    result = torch.clamp(result_quantized + output_zero_point, -128, 127).to(torch.int8)
    return result


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
