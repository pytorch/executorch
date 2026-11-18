/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>

#include <sdkconfig.h>

#include <esp_nn.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace espressif::native {

using executorch::aten::IntArrayRef;
using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;

// cppcheck-suppress unusedFunction
Tensor& quantized_depthwise_conv2d_int8_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    int64_t channel_multiplier,
    const Tensor& output_multiplier,
    const Tensor& output_shift,
    int64_t input_offset,
    int64_t output_offset,
    const Tensor& scratch,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      input.dim() == 4 && weight.dim() == 4 && out.dim() == 4 &&
          stride.size() == 2 && padding.size() == 2,
      InvalidArgument,
      out);
  const auto batches = input.size(0);
  const auto height = input.size(1);
  const auto width = input.size(2);
  const auto channels = input.size(3);
  const auto kernel_h = weight.size(1);
  const auto kernel_w = weight.size(2);
  const auto output_h = out.size(1);
  const auto output_w = out.size(2);
  const int64_t dimensions[] = {
      height,
      width,
      channels,
      kernel_h,
      kernel_w,
      output_h,
      output_w,
      stride[0],
      stride[1],
      channel_multiplier};
  for (const auto value : dimensions) {
    ET_KERNEL_CHECK(
        ctx,
        value > 0 && value <= std::numeric_limits<uint16_t>::max(),
        InvalidArgument,
        out);
  }
  for (size_t axis = 0; axis < 2; ++axis) {
    ET_KERNEL_CHECK(
        ctx,
        padding[axis] >= 0 && padding[axis] <= 32768 &&
            (out.size(axis + 1) - 1) * stride[axis] - padding[axis] <= 32767 &&
            (out.size(axis + 1) - 1) * stride[axis] + weight.size(axis + 1) <=
                input.size(axis + 1) + 2 * padding[axis],
        InvalidArgument,
        out);
  }
  const uint64_t outputs = uint64_t(channels) * channel_multiplier;
  const uint64_t aligned_channels = (uint64_t(channels) + 15) & ~uint64_t(15);
  const uint64_t input_bytes = uint64_t(height) * width * channels;
  const uint64_t output_bytes = uint64_t(output_h) * output_w * outputs;
  const uint64_t filter_bytes = uint64_t(kernel_h) * kernel_w * outputs;
  const uint64_t padded_h =
      height + 2 * padding[0] + 2 * kernel_h + 2 * stride[0];
  const uint64_t padded_w =
      width + 2 * padding[1] + 2 * kernel_w + 2 * stride[1];
  const uint64_t scratch_bound =
      2 * uint64_t(kernel_h) * kernel_w * std::max(aligned_channels, outputs) +
      2 * padded_h * padded_w * aligned_channels +
      uint64_t(output_h) * output_w * aligned_channels + 12 * aligned_channels +
      128;
  constexpr auto limit = std::numeric_limits<int32_t>::max();
  ET_KERNEL_CHECK(
      ctx,
      outputs <= 65535 && aligned_channels <= 65535 && scratch_bound <= limit &&
          input_bytes <= limit && output_bytes <= limit && batches >= 0 &&
          filter_bytes <= weight.nbytes() &&
          uint64_t(batches) <= input.nbytes() / input_bytes &&
          uint64_t(batches) <= out.nbytes() / output_bytes &&
          outputs <= output_multiplier.nbytes() / sizeof(int32_t) &&
          outputs <= output_shift.nbytes() / sizeof(int32_t) &&
          (!bias || outputs <= bias->nbytes() / sizeof(int32_t)),
      InvalidArgument,
      out);

  const data_dims_t input_dims{
      int32_t(width), int32_t(height), int32_t(channels), 1};
  const data_dims_t filter_dims{
      int32_t(kernel_w), int32_t(kernel_h), int32_t(outputs), 1};
  const data_dims_t output_dims{
      int32_t(output_w), int32_t(output_h), int32_t(outputs), 1};
  const dw_conv_params_t params{
      int32_t(input_offset),
      int32_t(output_offset),
      int32_t(channel_multiplier),
      {int32_t(stride[1]), int32_t(stride[0])},
      {int32_t(padding[1]), int32_t(padding[0])},
      {1, 1},
      {-128, 127}};
  const quant_data_t quant{
      const_cast<int32_t*>(output_shift.const_data_ptr<int32_t>()),
      const_cast<int32_t*>(output_multiplier.const_data_ptr<int32_t>())};
  const auto required = esp_nn_get_depthwise_conv_scratch_size(
      &input_dims, &filter_dims, &output_dims, &params);
  ET_KERNEL_CHECK(
      ctx,
      required >= 0 && size_t(required) <= scratch.nbytes(),
      InvalidArgument,
      out);
  esp_nn_set_depthwise_conv_scratch_buf(scratch.mutable_data_ptr<uint8_t>());
  for (int64_t batch = 0; batch < batches; ++batch) {
    esp_nn_depthwise_conv_s8(
        &input_dims,
        input.const_data_ptr<int8_t>() + batch * input_bytes,
        &filter_dims,
        weight.const_data_ptr<int8_t>(),
        bias ? bias->const_data_ptr<int32_t>() : nullptr,
        &output_dims,
        out.mutable_data_ptr<int8_t>() + batch * output_bytes,
        &params,
        &quant);
  }
  esp_nn_set_depthwise_conv_scratch_buf(nullptr);
  return out;
}

} // namespace espressif::native
