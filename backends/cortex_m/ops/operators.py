# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.cortex_m.passes.passes_utils import (
    dequantize_per_tensor_cmsis,
    quantize_per_tensor_cmsis,
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
    self_fp = dequantize_per_tensor_cmsis(
        self, self_zero_point, self_multiplier, self_shift
    )
    other_fp = dequantize_per_tensor_cmsis(
        other, other_zero_point, other_multiplier, other_shift
    )
    result_fp = self_fp + other_fp
    result_quantized = quantize_per_tensor_cmsis(
        result_fp, output_zero_point, output_multiplier, output_shift
    )
    return result_quantized


# Define the operator schema with multipliers and shifts (11 args + out tensor)
lib.define(
    "quantized_add.out("
    "Tensor self, Scalar self_zero_point, Scalar self_multiplier, Scalar self_shift, "
    "Tensor other, Scalar other_zero_point, Scalar other_multiplier, Scalar other_shift, "
    "Scalar output_zero_point, Scalar output_multiplier, Scalar output_shift, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)


# Fake meta function for shape and dtype inference during compilation
@register_fake("cortex_m::quantized_add.out")
def quantized_add_out_meta(
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
    out: torch.Tensor,
) -> torch.Tensor:
    # Validate against correct broadcasted shape
    expected_shape = torch.broadcast_shapes(self.shape, other.shape)
    assert (
        out.shape == expected_shape
    ), f"Output shape {out.shape} must match broadcasted shape {expected_shape}"
    return out


# Actual implementation delegating to backend or custom kernel
@impl(lib, "quantized_add.out", "CompositeExplicitAutograd")
def quantized_add_out_impl(
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
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    self_fp = dequantize_per_tensor_cmsis(
        self, self_zero_point, self_multiplier, self_shift
    )
    other_fp = dequantize_per_tensor_cmsis(
        other, other_zero_point, other_multiplier, other_shift
    )
    result_fp = self_fp + other_fp
    result_quantized = quantize_per_tensor_cmsis(
        result_fp, output_zero_point, output_multiplier, output_shift
    )

    # Write into the provided output tensor
    out.copy_(result_quantized)

    return out


# ===================================================================
# QUANTIZED LINEAR OPERATION DEFINITION
# ===================================================================


def _check_per_tensor_or_per_channel(param: torch.Tensor, out_channels: int, name: str):
    assert param.numel() in [
        1,
        out_channels,
    ], f"{name} must be per-tensor (1) or per-channel ({out_channels}), got {param.numel()}"


lib.define(
    "quantized_linear.out("
    "Tensor input, Scalar input_zero_point, Scalar input_multiplier, Scalar input_shift, "
    "Tensor weights, "
    "Tensor weight_zero_point, Tensor weight_multiplier, Tensor weight_shift, "
    "Tensor? bias, Tensor bias_multiplier, Tensor bias_shift, "
    "Tensor scratch_buffer, Scalar output_zero_point, Scalar in_features, Scalar out_features, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)

# Define functional variant (non-out version)
lib.define(
    "quantized_linear("
    "Tensor input, Scalar input_zero_point, Scalar input_multiplier, Scalar input_shift, "
    "Tensor weights, "
    "Tensor weight_zero_point, Tensor weight_multiplier, Tensor weight_shift, "
    "Tensor? bias, Tensor bias_multiplier, Tensor bias_shift, "
    "Tensor scratch_buffer, Scalar output_zero_point, Scalar in_features, Scalar out_features"
    ") -> Tensor"
)


# Fake meta function for shape inference (out variant)
@register_fake("cortex_m::quantized_linear.out")
def quantized_linear_out_meta(
    input: torch.Tensor,
    input_zero_point: int,
    input_multiplier: int,
    input_shift: int,
    weights: torch.Tensor,
    weight_zero_point: torch.Tensor,
    weight_multiplier: torch.Tensor,
    weight_shift: torch.Tensor,
    bias: torch.Tensor,
    bias_multiplier: torch.Tensor,
    bias_shift: torch.Tensor,
    scratch_buffer: torch.Tensor,
    output_zero_point: int,
    in_features: int,
    out_features: int,
    out: torch.Tensor,
) -> torch.Tensor:
    # Validate dimensions
    batch_size = input.shape[0]
    out_channels = weights.shape[0]

    # Validate weight quantization parameters dimensions
    _check_per_tensor_or_per_channel(
        weight_zero_point, out_channels, "weight_zero_point"
    )
    _check_per_tensor_or_per_channel(
        weight_multiplier, out_channels, "weight_multiplier"
    )
    _check_per_tensor_or_per_channel(weight_shift, out_channels, "weight_shift")

    # Validate output shape
    expected_shape = (batch_size, out_channels)
    assert (
        out.shape == expected_shape
    ), f"Output shape {out.shape} must be {expected_shape}"

    return out


# Fake meta function for shape inference (functional variant)
@register_fake("cortex_m::quantized_linear")
def quantized_linear_meta(
    input: torch.Tensor,
    input_zero_point: int,
    input_multiplier: int,
    input_shift: int,
    weights: torch.Tensor,
    weight_zero_point: torch.Tensor,
    weight_multiplier: torch.Tensor,
    weight_shift: torch.Tensor,
    bias: torch.Tensor,
    bias_multiplier: torch.Tensor,
    bias_shift: torch.Tensor,
    scratch_buffer: torch.Tensor,
    output_zero_point: int,
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    # Validate dimensions (same as out variant)
    batch_size = input.shape[0]
    out_channels = weights.shape[0]

    # Validate weight quantization parameters dimensions
    _check_per_tensor_or_per_channel(
        weight_zero_point, out_channels, "weight_zero_point"
    )
    _check_per_tensor_or_per_channel(
        weight_multiplier, out_channels, "weight_multiplier"
    )
    _check_per_tensor_or_per_channel(weight_shift, out_channels, "weight_shift")

    # Calculate output shape for functional variant
    output_shape = (batch_size, out_channels)
    return torch.empty(output_shape, dtype=input.dtype, device=input.device)


@impl(lib, "quantized_linear.out", "CompositeExplicitAutograd")
def quantized_linear_out_impl(
    input: torch.Tensor,
    input_zero_point: int,
    input_multiplier: int,
    input_shift: int,
    weights: torch.Tensor,
    weight_zero_point: torch.Tensor,
    weight_multiplier: torch.Tensor,
    weight_shift: torch.Tensor,
    bias: torch.Tensor,
    bias_multiplier: torch.Tensor,
    bias_shift: torch.Tensor,
    scratch_buffer: torch.Tensor,
    output_zero_point: int,
    in_features: int,
    out_features: int,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    """
    Fallback implementation for meta/testing
    Note: This won't be called at runtime, only during compilation
    """

    # Per-channel dequantization
    input_scale = input_multiplier * (2.0 ** (-input_shift))
    input_fp = (input.float() - input_zero_point) * input_scale
    if weight_zero_point.numel() == 1:
        # Per-tensor
        weight_scale = weight_multiplier.item() * (2.0 ** (-weight_shift.item()))
        weights_fp = (weights.float() - weight_zero_point.item()) * weight_scale
    else:
        # Per-channel
        weight_scales = weight_multiplier.float() * (2.0 ** (-weight_shift.float()))
        weights_fp = (
            weights.float() - weight_zero_point.float().unsqueeze(1)
        ) * weight_scales.unsqueeze(1)
    bias_fp = None
    if bias is not None:
        bias_scales = bias_multiplier.float() * (2.0 ** (-bias_shift.float()))
        bias_fp = bias.float() * bias_scales

        result_fp = torch.nn.functional.linear(input_fp, weights_fp, bias_fp)
    else:
        result_fp = torch.nn.functional.linear(input_fp, weights_fp)
    result_quantized = torch.clamp(
        torch.round(result_fp + output_zero_point), -128, 127
    ).to(torch.int8)
    out.copy_(result_quantized)
    return out


# Functional variant implementation
@impl(lib, "quantized_linear", "CompositeExplicitAutograd")
def quantized_linear_impl(
    input: torch.Tensor,
    input_zero_point: int,
    input_multiplier: int,
    input_shift: int,
    weights: torch.Tensor,
    weight_zero_point: torch.Tensor,
    weight_multiplier: torch.Tensor,
    weight_shift: torch.Tensor,
    bias: torch.Tensor,
    bias_multiplier: torch.Tensor,
    bias_shift: torch.Tensor,
    scratch_buffer: torch.Tensor,
    output_zero_point: int,
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    """
    Functional variant - creates output tensor and calls out variant
    """
    # Create output tensor
    batch_size = input.shape[0]
    output = torch.empty(
        (batch_size, out_features), dtype=torch.int8, device=input.device
    )
    return quantized_linear_out_impl(
        input,
        input_zero_point,
        input_multiplier,
        input_shift,
        weights,
        weight_zero_point,
        weight_multiplier,
        weight_shift,
        bias,
        bias_multiplier,
        bias_shift,
        scratch_buffer,
        output_zero_point,
        in_features,
        out_features,
        out=output,
    )
