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
Tensor& quantized_mul_int8_out(
    KernelRuntimeContext& ctx,
    const Tensor& input1,
    const Tensor& input2,
    int64_t input1_offset,
    int64_t input2_offset,
    int64_t output_offset,
    int64_t output_multiplier,
    int64_t output_shift,
    Tensor& out) {
  const auto numel = input1.numel();
  ET_KERNEL_CHECK(
      ctx,
      numel >= 0 && numel <= std::numeric_limits<int32_t>::max(),
      InvalidArgument,
      out);
  const auto size = static_cast<size_t>(numel);
  ET_KERNEL_CHECK(
      ctx,
      size <= input1.nbytes() && size <= input2.nbytes() &&
          size <= out.nbytes(),
      InvalidArgument,
      out);

  esp_nn_mul_elementwise_s8(
      input1.const_data_ptr<int8_t>(),
      input2.const_data_ptr<int8_t>(),
      static_cast<int32_t>(input1_offset),
      static_cast<int32_t>(input2_offset),
      out.mutable_data_ptr<int8_t>(),
      static_cast<int32_t>(output_offset),
      static_cast<int32_t>(output_multiplier),
      static_cast<int32_t>(output_shift),
      std::numeric_limits<int8_t>::min(),
      std::numeric_limits<int8_t>::max(),
      static_cast<int32_t>(numel));
  return out;
}

} // namespace espressif::native
