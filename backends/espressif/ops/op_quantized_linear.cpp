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
Tensor& quantized_linear_int8_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const Tensor& output_multiplier,
    const Tensor& output_shift,
    int64_t input_offset,
    int64_t output_offset,
    int64_t activation_min,
    int64_t activation_max,
    Tensor& out) {
  ET_KERNEL_CHECK(ctx, weight.dim() >= 2, InvalidArgument, out);
  const auto channels = weight.size(0);
  const auto columns = weight.size(1);
  ET_KERNEL_CHECK(
      ctx,
      channels > 0 && channels <= std::numeric_limits<uint16_t>::max() &&
          columns > 0 && columns <= std::numeric_limits<uint16_t>::max(),
      InvalidArgument,
      out);
  const auto batches = input.numel() / columns;
  ET_KERNEL_CHECK(
      ctx,
      batches >= 0 && batches <= std::numeric_limits<int32_t>::max(),
      InvalidArgument,
      out);
  const auto rows = static_cast<size_t>(batches);
  const auto cols = static_cast<size_t>(columns);
  const auto outputs = static_cast<size_t>(channels);
  ET_KERNEL_CHECK(
      ctx,
      rows <= input.nbytes() / cols && rows <= out.nbytes() / outputs &&
          outputs <= weight.nbytes() / cols &&
          outputs <= output_multiplier.nbytes() / sizeof(int32_t) &&
          outputs <= output_shift.nbytes() / sizeof(int32_t) &&
          (!bias || outputs <= bias->nbytes() / sizeof(int32_t)),
      InvalidArgument,
      out);

  esp_nn_fully_connected_per_ch_s8_batch(
      input.const_data_ptr<int8_t>(),
      static_cast<int32_t>(input_offset),
      static_cast<uint16_t>(columns),
      weight.const_data_ptr<int8_t>(),
      0,
      bias ? bias->const_data_ptr<int32_t>() : nullptr,
      out.mutable_data_ptr<int8_t>(),
      static_cast<uint16_t>(channels),
      static_cast<int32_t>(output_offset),
      output_shift.const_data_ptr<int32_t>(),
      output_multiplier.const_data_ptr<int32_t>(),
      static_cast<int32_t>(activation_min),
      static_cast<int32_t>(activation_max),
      static_cast<int32_t>(batches));
  return out;
}

} // namespace espressif::native
