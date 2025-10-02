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

namespace impl {
namespace HiFi {
namespace native {

using ::executorch::aten::Tensor;
using ::executorch::runtime::getLeadingDims;
using ::executorch::runtime::KernelRuntimeContext;
using std::optional;

void quantized_linear_asym8sxasym8s_asym8s_per_tensor_out(
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
  const int64_t leading_dims = getLeadingDims(in, in.dim() - 1);
  const int64_t out_dim = weight.size(0); // = out_dim
  const int64_t in_dim = weight.size(1); // = in_dim

  const int8_t* __restrict__ in_data = in.const_data_ptr<int8_t>();
  const int8_t* __restrict__ weight_data = weight.const_data_ptr<int8_t>();
  const int32_t* __restrict__ bias_data = bias.const_data_ptr<int32_t>();
  int8_t* __restrict__ out_data = out.mutable_data_ptr<int8_t>();

  const int32_t out_multipler_int32 = static_cast<int32_t>(out_multiplier);
  const int32_t out_shift_int32 = static_cast<int32_t>(out_shift);

  // The nnlib kernel to compute quantized linear via matmul.
  const int32_t ret = xa_nn_matmul_asym8sxasym8s_asym8s(
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
      -weight_zero_point, // mat1_zero_bias
      -in_zero_point, // mat2_zero_bias
      out_multipler_int32, // out_multiplier
      out_shift_int32, // out_shift
      out_zero_point); // out_zero_bias
  ET_DCHECK_MSG(ret == 0, "HiFi quantized::linear_per_tensor failed");
}

} // namespace native
} // namespace HiFi
} // namespace impl
