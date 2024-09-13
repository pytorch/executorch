/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/reference/kernels/kernels.h>

#include <executorch/runtime/kernel/kernel_includes.h>
#include <algorithm>
#include <cmath>

namespace impl {
namespace HiFi {
namespace native {

using Tensor = exec_aten::Tensor;
using executorch::runtime::KernelRuntimeContext;

void quantized_linear_out(
    KernelRuntimeContext& ctx,
    const Tensor& src,
    const Tensor& weight,
    const Tensor& bias,
    int64_t src_zero_point,
    const Tensor& weight_zero_point,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    int64_t out_zero_point,
    const exec_aten::optional<Tensor>& offset,
    Tensor& out) {
  // input comes in shape [leading_dims, in_dim]
  // weight comes in shape [out_dim, in_dim]
  // output comes in empty with shape [leading_dims, out_dim]
  // Perform matrix multiply (M x N) x (N x P)' => M x P
  int64_t leading_dims = getLeadingDims(src, src.dim() - 1);
  int64_t out_dim = weight.size(0); // = out_dim
  int64_t in_dim = weight.size(1); // = in_dim

  const uint8_t* __restrict__ in_data = src.const_data_ptr<uint8_t>();
  const uint8_t* __restrict__ weight_data = weight.const_data_ptr<uint8_t>();
  const int32_t* __restrict__ bias_data = bias.const_data_ptr<int32_t>();
  uint8_t* __restrict__ out_data = out.mutable_data_ptr<uint8_t>();

  // The nnlib kernel to compute quantized linear via matmul.
  int32_t ret = impl::HiFi::kernels::matmul_asym8uxasym8u_asym8u(
      out_data, // p_out
      weight_data, // p_mat1,
      in_data, // p_mat2,
      bias_data, // p_bias
      out_dim, // rows of p_mat1
      in_dim, // cols of p_mat1
      in_dim, // row_stride of p_mat1
      leading_dims, // vec_count, i.e., rows of p_mat2
      in_dim, // vec_offset of p_mat2.
      out_dim, // out_offset, i.e., offset of next output element written
      1, // out_stride, i.e., stride to go to next output row
      -weight_zero_point.const_data_ptr<int32_t>()[0], // mat1_zero_bias
      -src_zero_point, // mat2_zero_bias
      out_multiplier.const_data_ptr<int32_t>(), // out_multiplier
      out_shift.const_data_ptr<int32_t>(), // out_shift
      out_zero_point, // out_zero_bias
      false); // per channel quantization
  ET_DCHECK_MSG(ret == 0, "HiFi quantized::linear failed");
}

}; // namespace native
}; // namespace HiFi
}; // namespace impl
