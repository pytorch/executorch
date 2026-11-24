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
Tensor& quantized_conv2d_int8_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
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
  const auto outputs = out.size(3);
  const int64_t dimensions[] = {
      height,
      width,
      channels,
      kernel_h,
      kernel_w,
      output_h,
      output_w,
      outputs,
      stride[0],
      stride[1]};
  for (const auto value : dimensions) {
    ET_KERNEL_CHECK(
        ctx,
        value > 0 && value <= std::numeric_limits<uint16_t>::max(),
        InvalidArgument,
        out);
  }
  for (const auto value : padding) {
    ET_KERNEL_CHECK(
        ctx,
        value >= 0 && value <= std::numeric_limits<uint16_t>::max(),
        InvalidArgument,
        out);
  }
  const uint64_t input_bytes = uint64_t(height) * width * channels;
  const uint64_t output_bytes = uint64_t(output_h) * output_w * outputs;
  const uint64_t filter_bytes =
      uint64_t(kernel_h) * kernel_w * channels * outputs;
  const uint64_t padded_h = std::max<int64_t>(
      height + padding[0], (output_h - 1) * stride[0] + kernel_h);
  const uint64_t padded_w = std::max<int64_t>(
      width + padding[1], (output_w - 1) * stride[1] + kernel_w);
  const uint64_t aligned_channels = (uint64_t(channels) + 15) & ~uint64_t(15);
  const uint64_t aligned_filter_bytes =
      (uint64_t(kernel_w) * aligned_channels + 15) * kernel_h * outputs;
  const uint64_t staging_bytes =
      40 * ((uint64_t(kernel_h) * kernel_w * channels + 15) & ~uint64_t(15));
  constexpr auto limit = std::numeric_limits<int32_t>::max();
  ET_KERNEL_CHECK(
      ctx,
      batches >= 0 && input_bytes <= limit && output_bytes <= limit &&
          filter_bytes <= weight.nbytes() && aligned_filter_bytes <= limit &&
          staging_bytes <= limit &&
          padded_h <= limit / (padded_w * aligned_channels) &&
          uint64_t(batches) <= input.nbytes() / input_bytes &&
          uint64_t(batches) <= out.nbytes() / output_bytes &&
          uint64_t(outputs) <= output_multiplier.nbytes() / sizeof(int32_t) &&
          uint64_t(outputs) <= output_shift.nbytes() / sizeof(int32_t) &&
          (!bias || uint64_t(outputs) <= bias->nbytes() / sizeof(int32_t)),
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(
      ctx,
      padded_h * padded_w * aligned_channels + aligned_filter_bytes +
              staging_bytes + 8 * outputs + 16 * 1024 + 256 <=
          limit,
      InvalidArgument,
      out);

  const data_dims_t input_dims{
      int32_t(width), int32_t(height), int32_t(channels), 1};
  const data_dims_t filter_dims{
      int32_t(kernel_w), int32_t(kernel_h), int32_t(channels), 1};
  const data_dims_t output_dims{
      int32_t(output_w), int32_t(output_h), int32_t(outputs), 1};
  const conv_params_t params{
      int32_t(input_offset),
      int32_t(output_offset),
      {int32_t(stride[1]), int32_t(stride[0])},
      {int32_t(padding[1]), int32_t(padding[0])},
      {1, 1},
      {-128, 127}};
  const quant_data_t quant{
      const_cast<int32_t*>(output_shift.const_data_ptr<int32_t>()),
      const_cast<int32_t*>(output_multiplier.const_data_ptr<int32_t>())};
  const auto required = esp_nn_get_conv_scratch_size(
      &input_dims, &filter_dims, &output_dims, &params);
  ET_KERNEL_CHECK(
      ctx,
      required >= 0 && size_t(required) <= scratch.nbytes(),
      InvalidArgument,
      out);
  esp_nn_set_conv_scratch_buf(scratch.mutable_data_ptr<uint8_t>());
  for (int64_t batch = 0; batch < batches; ++batch) {
    esp_nn_conv_s8(
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
  esp_nn_set_conv_scratch_buf(nullptr);
  return out;
}

} // namespace espressif::native
