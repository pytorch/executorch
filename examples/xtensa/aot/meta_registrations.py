# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.scalar_type import ScalarType
from torch.library import impl, Library

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
    "quantized_linear_pt2(Tensor src, Tensor weight, Tensor bias, float src_scale, int src_zero_point, float weight_scale, int weight_zero_point, Tensor out_multiplier, Tensor out_shift, int out_zero_point) -> (Tensor Z)"
)
lib.define(
    "quantized_linear_pt2.out(Tensor src, Tensor weight, Tensor bias, float src_scale, int src_zero_point, float weight_scale, int weight_zero_point, Tensor out_multiplier, Tensor out_shift, int out_zero_point, *, Tensor(a!) out) ->  Tensor(a!)"
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


@impl(m, "quantized_linear_pt2")
def quantized_linear_pt2_meta(
    src: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    in_scale: float,
    in_zero_point: int,
    weight_scale: float,
    weight_zero_point: int,
    out_multiplier: int,
    out_shift: int,
    out_zero_point: int,
):
    # src comes in shape [leading_dims, in_dim]
    # weight comes in shape [out_dim, in_dim]
    # output comes in empty with shape [leading_dims, out_dim]
    out_size = list(src.size())
    weight_size = list(weight.size())
    assert len(weight_size) == 2
    out_size[-1] = weight_size[0]
    return src.new_empty(out_size, dtype=torch.uint8)
