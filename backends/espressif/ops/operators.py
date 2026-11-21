# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.espressif.utils.quantization import requantize
from torch.library import impl, Library, register_fake

lib = Library("esp_nn", "DEF")
lib.define("quantized_lut_int8(Tensor input, Tensor table) -> Tensor")
lib.define(
    "quantized_lut_int8.out(Tensor input, Tensor table, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_mul_int8(Tensor input1, Tensor input2, int input1_offset, "
    "int input2_offset, int output_offset, int output_multiplier, int output_shift, "
    "int activation_min, int activation_max) -> Tensor"
)
lib.define(
    "quantized_mul_int8.out(Tensor input1, Tensor input2, int input1_offset, "
    "int input2_offset, int output_offset, int output_multiplier, int output_shift, "
    "int activation_min, int activation_max, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_mul_broadcast_int8(Tensor input1, Tensor input2, int input1_offset, "
    "int input2_offset, int output_offset, int output_multiplier, int output_shift, "
    "int activation_min, int activation_max) -> Tensor"
)
lib.define(
    "quantized_mul_broadcast_int8.out(Tensor input1, Tensor input2, int input1_offset, "
    "int input2_offset, int output_offset, int output_multiplier, int output_shift, "
    "int activation_min, int activation_max, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_add_int8(Tensor input1, Tensor input2, int input1_offset, "
    "int input2_offset, int output_offset, int input1_multiplier, int input1_shift, "
    "int input2_multiplier, int input2_shift, int left_shift, "
    "int output_multiplier, int output_shift, "
    "int activation_min, int activation_max) -> Tensor"
)
lib.define(
    "quantized_add_int8.out(Tensor input1, Tensor input2, int input1_offset, "
    "int input2_offset, int output_offset, int input1_multiplier, int input1_shift, "
    "int input2_multiplier, int input2_shift, int left_shift, "
    "int output_multiplier, int output_shift, int activation_min, int activation_max, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)


lib.define(
    "quantized_max_pool2d_int8(Tensor input, int[2] kernel_size, int[2] stride, "
    "int[2] padding) -> Tensor"
)
lib.define(
    "quantized_max_pool2d_int8.out(Tensor input, int[2] kernel_size, int[2] stride, "
    "int[2] padding, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_avg_pool2d_int8(Tensor input, int[2] kernel_size, int[2] stride, "
    "int[2] padding) -> Tensor"
)
lib.define(
    "quantized_avg_pool2d_int8.out(Tensor input, int[2] kernel_size, int[2] stride, "
    "int[2] padding, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_linear_int8(Tensor input, Tensor weight, Tensor? bias, "
    "Tensor output_multiplier, Tensor output_shift, int input_offset, int output_offset, "
    "int activation_min, int activation_max) -> Tensor"
)
lib.define(
    "quantized_linear_int8.out(Tensor input, Tensor weight, Tensor? bias, "
    "Tensor output_multiplier, Tensor output_shift, int input_offset, int output_offset, "
    "int activation_min, int activation_max, *, Tensor(a!) out) -> Tensor(a!)"
)

lib.define(
    "quantized_conv2d_int8(Tensor input, Tensor weight, Tensor? bias, "
    "int[2] stride, int[2] padding, Tensor output_multiplier, Tensor output_shift, "
    "int input_offset, int output_offset, int activation_min, int activation_max, "
    "Tensor scratch) -> Tensor"
)
lib.define(
    "quantized_conv2d_int8.out(Tensor input, Tensor weight, Tensor? bias, "
    "int[2] stride, int[2] padding, Tensor output_multiplier, Tensor output_shift, "
    "int input_offset, int output_offset, int activation_min, int activation_max, "
    "Tensor scratch, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "quantized_depthwise_conv2d_int8(Tensor input, Tensor weight, Tensor? bias, "
    "int[2] stride, int[2] padding, int channel_multiplier, "
    "Tensor output_multiplier, Tensor output_shift, "
    "int input_offset, int output_offset, int activation_min, int activation_max, "
    "Tensor scratch) -> Tensor"
)
lib.define(
    "quantized_depthwise_conv2d_int8.out(Tensor input, Tensor weight, Tensor? bias, "
    "int[2] stride, int[2] padding, int channel_multiplier, "
    "Tensor output_multiplier, Tensor output_shift, "
    "int input_offset, int output_offset, int activation_min, int activation_max, "
    "Tensor scratch, *, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("esp_nn::quantized_lut_int8")
def quantized_lut_int8_meta(input, table):
    return input.new_empty(input.shape)


@impl(lib, "quantized_lut_int8", "CompositeExplicitAutograd")
def quantized_lut_int8(input, table):
    return torch.take(table, input.to(torch.int64) & 255)


@register_fake("esp_nn::quantized_max_pool2d_int8")
@register_fake("esp_nn::quantized_avg_pool2d_int8")
def quantized_pool2d_int8_meta(input, kernel_size, stride, padding):
    batch, height, width, channels = input.shape
    output_height = (height + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    output_width = (width + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    return input.new_empty((batch, output_height, output_width, channels))


@impl(lib, "quantized_max_pool2d_int8", "CompositeExplicitAutograd")
def quantized_max_pool2d_int8(input, kernel_size, stride, padding):
    result = torch.nn.functional.max_pool2d(
        input.permute(0, 3, 1, 2).to(torch.float32), kernel_size, stride, padding
    )
    return result.permute(0, 2, 3, 1).to(torch.int8).contiguous()


@impl(lib, "quantized_avg_pool2d_int8", "CompositeExplicitAutograd")
def quantized_avg_pool2d_int8(input, kernel_size, stride, padding):
    result = quantized_pool2d_int8_meta(input, kernel_size, stride, padding)
    for y in range(result.shape[1]):
        top = y * stride[0] - padding[0]
        bottom = min(top + kernel_size[0], input.shape[1])
        top = max(top, 0)
        for x in range(result.shape[2]):
            left = x * stride[1] - padding[1]
            right = min(left + kernel_size[1], input.shape[2])
            left = max(left, 0)
            count = (bottom - top) * (right - left)
            total = input[:, top:bottom, left:right, :].to(torch.int64).sum((1, 2))
            nudge = torch.where(total > 0, count // 2, -(count // 2))
            result[:, y, x, :] = (
                torch.div(total + nudge, count, rounding_mode="trunc")
                .clamp(-128, 127)
                .to(torch.int8)
            )
    return result


@register_fake("esp_nn::quantized_mul_int8")
def quantized_mul_int8_meta(
    input1,
    input2,
    input1_offset,
    input2_offset,
    output_offset,
    output_multiplier,
    output_shift,
    activation_min,
    activation_max,
):
    return torch.empty_like(input1)


@register_fake("esp_nn::quantized_mul_broadcast_int8")
def quantized_mul_broadcast_int8_meta(
    input1,
    input2,
    input1_offset,
    input2_offset,
    output_offset,
    output_multiplier,
    output_shift,
    activation_min,
    activation_max,
):
    return input1.new_empty(torch.broadcast_shapes(input1.shape, input2.shape))


@impl(lib, "quantized_mul_int8", "CompositeExplicitAutograd")
@impl(lib, "quantized_mul_broadcast_int8", "CompositeExplicitAutograd")
def quantized_mul_int8(
    input1,
    input2,
    input1_offset,
    input2_offset,
    output_offset,
    output_multiplier,
    output_shift,
    activation_min,
    activation_max,
):
    product = (input1.to(torch.int32) + input1_offset) * (
        input2.to(torch.int32) + input2_offset
    )
    result = requantize(product, output_multiplier, output_shift) + output_offset
    return result.clamp(activation_min, activation_max).to(torch.int8)


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
    activation_min,
    activation_max,
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
    activation_min,
    activation_max,
):
    lhs = (input1.to(torch.int32) + input1_offset) << left_shift
    rhs = (input2.to(torch.int32) + input2_offset) << left_shift
    total = requantize(lhs, input1_multiplier, input1_shift) + requantize(
        rhs, input2_multiplier, input2_shift
    )
    result = requantize(total, output_multiplier, output_shift) + output_offset
    return result.clamp(activation_min, activation_max).to(torch.int8)


@register_fake("esp_nn::quantized_linear_int8")
def quantized_linear_int8_meta(
    input,
    weight,
    bias,
    output_multiplier,
    output_shift,
    input_offset,
    output_offset,
    activation_min,
    activation_max,
):
    return input.new_empty((*input.shape[:-1], weight.shape[0]))


@impl(lib, "quantized_linear_int8", "CompositeExplicitAutograd")
def quantized_linear_int8(
    input,
    weight,
    bias,
    output_multiplier,
    output_shift,
    input_offset,
    output_offset,
    activation_min,
    activation_max,
):
    accumulator = (input.to(torch.int64) + input_offset) @ weight.to(torch.int64).T
    if bias is not None:
        accumulator = accumulator + bias
    result = torch.stack(
        [
            requantize(accumulator[..., channel], int(multiplier), int(shift))
            for channel, (multiplier, shift) in enumerate(
                zip(output_multiplier, output_shift)
            )
        ],
        dim=-1,
    )
    return (result + output_offset).clamp(activation_min, activation_max).to(torch.int8)


@register_fake("esp_nn::quantized_conv2d_int8")
def quantized_conv2d_int8_meta(
    input,
    weight,
    bias,
    stride,
    padding,
    output_multiplier,
    output_shift,
    input_offset,
    output_offset,
    activation_min,
    activation_max,
    scratch,
):
    batch, height, width, _ = input.shape
    outputs, kernel_h, kernel_w, _ = weight.shape
    return input.new_empty(
        (
            batch,
            (height + 2 * padding[0] - kernel_h) // stride[0] + 1,
            (width + 2 * padding[1] - kernel_w) // stride[1] + 1,
            outputs,
        )
    )


@impl(lib, "quantized_conv2d_int8", "CompositeExplicitAutograd")
def quantized_conv2d_int8(
    input,
    weight,
    bias,
    stride,
    padding,
    output_multiplier,
    output_shift,
    input_offset,
    output_offset,
    activation_min,
    activation_max,
    scratch,
):
    centered = input.to(torch.int64).permute(0, 3, 1, 2) + input_offset
    padded = torch.nn.functional.pad(
        centered, (padding[1], padding[1], padding[0], padding[0])
    )
    windows = padded.unfold(2, weight.shape[1], stride[0]).unfold(
        3, weight.shape[2], stride[1]
    )
    windows = windows.permute(0, 2, 3, 4, 5, 1).flatten(3)
    return quantized_linear_int8(
        windows,
        weight.flatten(1),
        bias,
        output_multiplier,
        output_shift,
        0,
        output_offset,
        activation_min,
        activation_max,
    )


@register_fake("esp_nn::quantized_depthwise_conv2d_int8")
def quantized_depthwise_conv2d_int8_meta(
    input,
    weight,
    bias,
    stride,
    padding,
    channel_multiplier,
    output_multiplier,
    output_shift,
    input_offset,
    output_offset,
    activation_min,
    activation_max,
    scratch,
):
    batch, height, width, channels = input.shape
    _, kernel_h, kernel_w, _ = weight.shape
    return input.new_empty(
        (
            batch,
            (height + 2 * padding[0] - kernel_h) // stride[0] + 1,
            (width + 2 * padding[1] - kernel_w) // stride[1] + 1,
            channels * channel_multiplier,
        )
    )


@impl(lib, "quantized_depthwise_conv2d_int8", "CompositeExplicitAutograd")
def quantized_depthwise_conv2d_int8(
    input,
    weight,
    bias,
    stride,
    padding,
    channel_multiplier,
    output_multiplier,
    output_shift,
    input_offset,
    output_offset,
    activation_min,
    activation_max,
    scratch,
):
    centered = input.to(torch.int64).permute(0, 3, 1, 2) + input_offset
    padded = torch.nn.functional.pad(
        centered, (padding[1], padding[1], padding[0], padding[0])
    )
    windows = padded.unfold(2, weight.shape[1], stride[0]).unfold(
        3, weight.shape[2], stride[1]
    )
    windows = windows.permute(0, 2, 3, 1, 4, 5).flatten(-2).unsqueeze(-2)
    filters = (
        weight[0].permute(2, 0, 1).reshape(input.shape[-1], channel_multiplier, -1)
    )
    accumulator = (windows * filters.to(torch.int64)).sum(-1).flatten(-2)
    if bias is not None:
        accumulator = accumulator + bias
    result = torch.stack(
        [
            requantize(accumulator[..., channel], int(multiplier), int(shift))
            for channel, (multiplier, shift) in enumerate(
                zip(output_multiplier, output_shift)
            )
        ],
        dim=-1,
    )
    return (result + output_offset).clamp(activation_min, activation_max).to(torch.int8)
