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
Tensor& quantized_avg_pool2d_int8_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      input.dim() == 4 && out.dim() == 4 && kernel_size.size() == 2 &&
          stride.size() == 2 && padding.size() == 2,
      InvalidArgument,
      out);
  const auto batches = input.size(0);
  const auto height = input.size(1);
  const auto width = input.size(2);
  const auto channels = input.size(3);
  const auto output_height = out.size(1);
  const auto output_width = out.size(2);
  for (const auto value :
       {height, width, channels, output_height, output_width}) {
    ET_KERNEL_CHECK(
        ctx,
        value > 0 && value <= std::numeric_limits<uint16_t>::max(),
        InvalidArgument,
        out);
  }
  for (size_t axis = 0; axis < 2; ++axis) {
    ET_KERNEL_CHECK(
        ctx,
        kernel_size[axis] > 0 &&
            kernel_size[axis] <= std::numeric_limits<uint16_t>::max() &&
            stride[axis] > 0 &&
            stride[axis] <= std::numeric_limits<uint16_t>::max() &&
            padding[axis] >= 0 && padding[axis] <= kernel_size[axis] / 2,
        InvalidArgument,
        out);
  }
  const auto window_count = std::min<int64_t>(height, kernel_size[0]) *
      std::min<int64_t>(width, kernel_size[1]);
  ET_KERNEL_CHECK(
      ctx,
      128 * window_count + window_count / 2 <=
          std::numeric_limits<int32_t>::max(),
      InvalidArgument,
      out);
  const auto input_bytes = static_cast<uint64_t>(height) * width * channels;
  const auto output_bytes =
      static_cast<uint64_t>(output_height) * output_width * channels;
  ET_KERNEL_CHECK(
      ctx,
      channels % 4 == 0 && batches >= 0 &&
          input_bytes <= std::numeric_limits<int32_t>::max() &&
          output_bytes <= std::numeric_limits<int32_t>::max() &&
          static_cast<size_t>(batches) <= input.nbytes() / input_bytes &&
          static_cast<size_t>(batches) <= out.nbytes() / output_bytes,
      InvalidArgument,
      out);
  for (int64_t batch = 0; batch < batches; ++batch) {
    esp_nn_avg_pool_s8(
        input.const_data_ptr<int8_t>() + batch * input_bytes,
        static_cast<uint16_t>(width),
        static_cast<uint16_t>(height),
        out.mutable_data_ptr<int8_t>() + batch * output_bytes,
        static_cast<uint16_t>(output_width),
        static_cast<uint16_t>(output_height),
        static_cast<uint16_t>(stride[1]),
        static_cast<uint16_t>(stride[0]),
        static_cast<uint16_t>(kernel_size[1]),
        static_cast<uint16_t>(kernel_size[0]),
        static_cast<uint16_t>(padding[1]),
        static_cast<uint16_t>(padding[0]),
        -128,
        127,
        static_cast<uint16_t>(channels));
  }
  return out;
}

} // namespace espressif::native
