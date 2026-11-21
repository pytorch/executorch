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
Tensor& quantized_lut_int8_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& table,
    Tensor& out) {
  const auto numel = input.numel();
  ET_KERNEL_CHECK(ctx, table.nbytes() >= 256, InvalidArgument, out);
  ET_KERNEL_CHECK(
      ctx,
      numel >= 0 && numel <= std::numeric_limits<int32_t>::max() &&
          static_cast<size_t>(numel) <= input.nbytes() &&
          static_cast<size_t>(numel) <= out.nbytes(),
      InvalidArgument,
      out);
  esp_nn_logistic_s8(
      input.const_data_ptr<int8_t>(),
      out.mutable_data_ptr<int8_t>(),
      static_cast<int32_t>(numel),
      table.const_data_ptr<int8_t>());
  return out;
}

} // namespace espressif::native
