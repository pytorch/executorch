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

#include <cstdint>
#include <limits>

namespace espressif::native {

using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;

// cppcheck-suppress unusedFunction
Tensor& quantized_mean_int8_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    int64_t input_zero_point,
    int64_t output_zero_point,
    int64_t output_multiplier,
    int64_t output_shift,
    Tensor& out) {
  ET_KERNEL_CHECK(ctx, input.dim() == 4, InvalidArgument, out);
  const auto batches = input.size(0);
  const auto height = input.size(1);
  const auto width = input.size(2);
  const auto channels = input.size(3);
  ET_KERNEL_CHECK(
      ctx,
      height > 0 && width > 0 && channels > 0 &&
          height <= std::numeric_limits<int32_t>::max() / width &&
          height * width <= std::numeric_limits<int32_t>::max() / channels,
      InvalidArgument,
      out);
  const auto count = height * width;
  const auto input_bytes = count * channels;
  ET_KERNEL_CHECK(
      ctx,
      count <= std::numeric_limits<int32_t>::max() / 255 && batches >= 0 &&
          static_cast<size_t>(batches) <= input.nbytes() / input_bytes &&
          static_cast<size_t>(batches) <= out.nbytes() / channels,
      InvalidArgument,
      out);
  for (int64_t batch = 0; batch < batches; ++batch) {
    esp_nn_mean_nhwc_s8(
        input.const_data_ptr<int8_t>() + batch * input_bytes,
        out.mutable_data_ptr<int8_t>() + batch * channels,
        static_cast<int32_t>(height),
        static_cast<int32_t>(width),
        static_cast<int32_t>(channels),
        static_cast<int32_t>(input_zero_point),
        static_cast<int32_t>(output_zero_point),
        static_cast<int32_t>(output_multiplier),
        static_cast<int32_t>(output_shift));
  }
  return out;
}

} // namespace espressif::native
