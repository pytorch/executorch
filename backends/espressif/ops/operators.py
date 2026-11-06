# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.espressif.utils.quantization import requantize
from torch.library import impl, Library, register_fake

lib = Library("esp_nn", "DEF")
lib.define(
    "quantized_mul_int8(Tensor input1, Tensor input2, int input1_offset, "
    "int input2_offset, int output_offset, int output_multiplier, int output_shift) -> Tensor"
)
lib.define(
    "quantized_mul_int8.out(Tensor input1, Tensor input2, int input1_offset, "
    "int input2_offset, int output_offset, int output_multiplier, int output_shift, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_add_int8(Tensor input1, Tensor input2, int input1_offset, "
    "int input2_offset, int output_offset, int input1_multiplier, int input1_shift, "
    "int input2_multiplier, int input2_shift, int left_shift, "
    "int output_multiplier, int output_shift) -> Tensor"
)
lib.define(
    "quantized_add_int8.out(Tensor input1, Tensor input2, int input1_offset, "
    "int input2_offset, int output_offset, int input1_multiplier, int input1_shift, "
    "int input2_multiplier, int input2_shift, int left_shift, "
    "int output_multiplier, int output_shift, *, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("esp_nn::quantized_mul_int8")
def quantized_mul_int8_meta(
    input1,
    input2,
    input1_offset,
    input2_offset,
    output_offset,
    output_multiplier,
    output_shift,
):
    return torch.empty_like(input1)


@impl(lib, "quantized_mul_int8", "CompositeExplicitAutograd")
def quantized_mul_int8(
    input1,
    input2,
    input1_offset,
    input2_offset,
    output_offset,
    output_multiplier,
    output_shift,
):
    product = (input1.to(torch.int32) + input1_offset) * (
        input2.to(torch.int32) + input2_offset
    )
    result = requantize(product, output_multiplier, output_shift) + output_offset
    return result.clamp(-128, 127).to(torch.int8)


@register_fake("esp_nn::quantized_add_int8")
def quantized_add_int8_meta(
    input1,
    input2,
    input1_offset,
    input2_offset,
    output_offset,
    input1_multiplier,
    input1_shift,
    input2_multiplier,
    input2_shift,
    left_shift,
    output_multiplier,
    output_shift,
):
    return torch.empty_like(input1)


@impl(lib, "quantized_add_int8", "CompositeExplicitAutograd")
def quantized_add_int8(
    input1,
    input2,
    input1_offset,
    input2_offset,
    output_offset,
    input1_multiplier,
    input1_shift,
    input2_multiplier,
    input2_shift,
    left_shift,
    output_multiplier,
    output_shift,
):
    lhs = (input1.to(torch.int32) + input1_offset) << left_shift
    rhs = (input2.to(torch.int32) + input2_offset) << left_shift
    total = requantize(lhs, input1_multiplier, input1_shift) + requantize(
        rhs, input2_multiplier, input2_shift
    )
    result = requantize(total, output_multiplier, output_shift) + output_offset
    return result.clamp(-128, 127).to(torch.int8)
