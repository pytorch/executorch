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

# Import these for the cadence function signatures.
import executorch.backends.cortex_m.cortex_m_ops_lib  # noqa: F401

###
# add.Tensor
###

lib.define(
    "add.Tensor(Tensor self, Tensor other, ScalarType dtype) -> (Tensor Z)"
)

lib.define(
    "add_Tensor.out(Tensor self, Tensor other, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)"
)

@impl(lib, "add.Tensor", "CompositeExplicitAutograd")
def aten_add_tensor_impl(
    input1: torch.Tensor,
    input2: torch.Tensor,
    dtype: torch.dtype,
    out: torch.Tensor,
) -> torch.Tensor:
    """
    The implementation of aten add.Tensor.
    """
    return exir_ops.edge.cortex_m.add.Tensor(input1, input2, dtype)

###
# add.out
###

lib.define(
    "add(Tensor input1, Tensor input2, ScalarType dtype) -> (Tensor Z)"
)

lib.define(
    "add.out(Tensor input1, Tensor input2, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)"
)

@impl(lib, "add.out", "CompositeExplicitAutograd")
def add_out_impl(
    input1: torch.Tensor,
    input2: torch.Tensor,
    dtype: torch.dtype,
    out: torch.Tensor,
) -> torch.Tensor:
    """
    The implementation of cmsis-nn add.out.
    """

    return exir_ops.edge.cortex_m.add.default(
        input1, input2, dtype, dtype
    )

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

lib.define(
    "softmax(Tensor self, int dim, bool half_to_float) -> Tensor"
)
lib.define(
    "softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)"
)
@impl(lib, "softmax", "CompositeExplicitAutograd")
def softmax_impl(self: torch.Tensor, dim: int, half_to_float: bool) -> torch.Tensor:
    # Call your custom edge op or fallback
    # return exir_ops.edge.cortex_m.softmax(self, dim, half_to_float)
    # ctx = get_kernel_ctx()  # gets KernelRuntimeContext*
    return {}
@impl(lib, "softmax.out", "CompositeExplicitAutograd")
def softmax_out_impl(self: torch.Tensor, dim: int, half_to_float: bool, out: torch.Tensor) -> torch.Tensor:
    return exir_ops.edge.cortex_m.softmax_out(self, dim, half_to_float, out)
