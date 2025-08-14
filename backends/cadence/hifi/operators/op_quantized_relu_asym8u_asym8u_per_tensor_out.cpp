/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <xa_nnlib_kernels_api.h>

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {

using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;

void quantized_relu_asym8u_asym8u_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const int64_t in_zero_point,
    const int64_t out_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    Tensor& output) {
  const uint8_t* __restrict__ input_data = input.const_data_ptr<uint8_t>();
  uint8_t* __restrict__ output_data = output.mutable_data_ptr<uint8_t>();

  const int32_t out_multipler_int32 = static_cast<int32_t>(out_multiplier);
  const int32_t out_shift_int32 = static_cast<int32_t>(out_shift);

  const int32_t ret = xa_nn_vec_relu_asym8u_asym8u(
      output_data,
      input_data,
      in_zero_point,
      out_multipler_int32,
      out_shift_int32,
      out_zero_point,
      0,
      255,
      input.numel());
  ET_DCHECK_MSG(
      ret == 0, "HiFi quantized_relu_asym8u_asym8u_per_tensor failed");
}

} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence
