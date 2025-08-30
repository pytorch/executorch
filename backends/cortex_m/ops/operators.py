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
