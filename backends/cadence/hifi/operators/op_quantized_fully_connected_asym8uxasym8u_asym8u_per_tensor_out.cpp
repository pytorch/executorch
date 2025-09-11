/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {

using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;
using std::optional;

void quantized_fully_connected_asym8uxasym8u_asym8u_per_tensor_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& in,
    const Tensor& weight,
    const Tensor& bias,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    int64_t out_multiplier,
    int64_t out_shift,
    int64_t out_zero_point,
    __ET_UNUSED const optional<Tensor>& offset,
    Tensor& out) {
  // input comes in shape [leading_dims, in_dim]
  // weight comes in shape [out_dim, in_dim]
  // output comes in empty with shape [leading_dims, out_dim]
  // Perform matrix multiply (M x N) x (N x P)' => M x P
  int64_t leading_dims = 1;
  int64_t out_dim = weight.size(0); // = out_dim
  int64_t in_dim = weight.size(1); // = in_dim

  const uint8_t* __restrict__ in_data = in.const_data_ptr<uint8_t>();
  const uint8_t* __restrict__ weight_data = weight.const_data_ptr<uint8_t>();
  const int32_t* __restrict__ bias_data = bias.const_data_ptr<int32_t>();
  uint8_t* __restrict__ out_data = out.mutable_data_ptr<uint8_t>();

  int32_t ret = xa_nn_fully_connected_asym8uxasym8u_asym8u(
      out_data,
      weight_data,
      in_data,
      bias_data,
      in_dim, // weight_depth, number of columns in weight
      out_dim, // out_depth, number of rows in weight
      -in_zero_point,
      -static_cast<int32_t>(weight_zero_point),
      static_cast<int32_t>(out_multiplier),
      static_cast<int32_t>(out_shift),
      out_zero_point);
  ET_DCHECK_MSG(ret == 0, "HiFi quantized::fully_connected failed");
}

} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence
