# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import (
    ops as exir_ops,
)  # To provide the implementation of the operators
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
