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
Tensor& quantized_softmax_int8_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    int64_t input_multiplier,
    int64_t input_shift,
    int64_t diff_min,
    const Tensor& scratch,
    Tensor& out) {
  ET_KERNEL_CHECK(ctx, input.dim() > 0, InvalidArgument, out);
  const auto width = input.size(input.dim() - 1);
  const auto count = input.numel();
  ET_KERNEL_CHECK(
      ctx,
      width > 0 && width <= 255 && count > 0 &&
          count <= std::numeric_limits<int32_t>::max() &&
          static_cast<size_t>(count) <= input.nbytes() &&
          static_cast<size_t>(count) <= out.nbytes(),
      InvalidArgument,
      out);
  const auto height = static_cast<int32_t>(count / width);
  const auto required = esp_nn_get_softmax_scratch_size(width, height);
  ET_KERNEL_CHECK(
      ctx,
      required >= 0 && size_t(required) <= scratch.nbytes(),
      InvalidArgument,
      out);
  esp_nn_set_softmax_scratch_buf(scratch.mutable_data_ptr<uint8_t>());
  esp_nn_softmax_s8(
      input.const_data_ptr<int8_t>(),
      height,
      width,
      static_cast<int32_t>(input_multiplier),
      static_cast<int32_t>(input_shift),
      static_cast<int32_t>(diff_min),
      out.mutable_data_ptr<int8_t>());
  esp_nn_set_softmax_scratch_buf(nullptr);
  return out;
}

} // namespace espressif::native
