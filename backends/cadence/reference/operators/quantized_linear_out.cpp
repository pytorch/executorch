/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/reference/kernels/kernels.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace impl {
namespace reference {
namespace native {

using executorch::aten::Tensor;
using executorch::runtime::getLeadingDims;
using executorch::runtime::KernelRuntimeContext;

void quantized_linear_out(
    KernelRuntimeContext& ctx,
    const Tensor& src,
    const Tensor& weight,
    const Tensor& bias,
    int64_t src_zero_point,
    const Tensor& weight_zero_point_t,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    int64_t out_zero_point,
    const exec_aten::optional<Tensor>& offset,
    Tensor& out) {
  // Assuming uint8_t for now, but needs to be updated for other quantization
  // types
  const uint8_t* __restrict__ src_data = src.const_data_ptr<uint8_t>();
  const uint8_t* __restrict__ weight_data = weight.const_data_ptr<uint8_t>();
  const int32_t* __restrict__ bias_data = bias.const_data_ptr<int32_t>();
  uint8_t* __restrict__ out_data = out.mutable_data_ptr<uint8_t>();

  int32_t weight_zero_point = weight_zero_point_t.const_data_ptr<int32_t>()[0];

  // input comes in shape [batch_size, in_dim]
  // weight comes in shape [out_dim, in_dim]
  // output comes in empty with shape [batch_size, out_dim]
  // Perform matrix multiply (M x N) x (N x P) => M x P
  const auto M = weight.size(0); // = out_dim
  const auto N = weight.size(1); // = in_dim

  // Given an N-dimensional input [d0, d1, d2, ..., d_{N-2}, d_{N-1}], the
  // leading dimensions is d0 * d1 * ... * d_{N-2}
  const auto leading_dims = getLeadingDims(src, src.dim() - 1);

  ET_CHECK_MSG(
      out_multiplier.numel() == 1, "out_multiplier should have one element");
  ET_CHECK_MSG(
      out_shift.numel() == 1, "out_multiplier should have one element");

  const int32_t* __restrict__ out_multiplier_data =
      out_multiplier.const_data_ptr<int32_t>();
  const int32_t* __restrict__ out_shift_data =
      out_shift.const_data_ptr<int32_t>();

  // Compute the out_scale from out_multiplier and out_shift
  const float out_scale =
      -out_multiplier_data[0] * 1.0 / (1 << 31) * pow(2, out_shift_data[0]);

  for (int i = 0; i < leading_dims; ++i) {
    for (int j = 0; j < M; ++j) {
      float sum = bias_data[j];
      for (int k = 0; k < N; ++k) {
        sum += (src_data[i * N + k] - src_zero_point) *
            (weight_data[j * N + k] - weight_zero_point);
      }
      out_data[i * M + j] =
          kernels::quantize<uint8_t>(sum, out_scale, out_zero_point);
    }
  }
}

}; // namespace native
}; // namespace reference
}; // namespace impl
