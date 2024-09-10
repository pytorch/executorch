/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/optimized/blas/CPUBlas.h>
#include <executorch/kernels/portable/cpu/util/matmul_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include <array>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

Tensor& opt_linear_out(
    RuntimeContext& ctx,
    const Tensor& in,
    const Tensor& mat2,
    const optional<Tensor>& bias,
    Tensor& out) {
  ET_KERNEL_CHECK_MSG(
      ctx,
      !bias.has_value(),
      InvalidArgument,
      out,
      "bias not supported yet in linear");
  ET_KERNEL_CHECK(ctx, check_linear_args(in, mat2, out), InvalidArgument, out);

  size_t output_ndim = 0;
  std::array<exec_aten::SizesType, kTensorDimensionLimit> output_sizes;
  get_linear_out_target_size(in, mat2, output_sizes.data(), &output_ndim);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {output_sizes.data(), output_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  // gemm on some platforms doesn't tolerate empty input.
  if (out.numel() == 0) {
    return out;
  }

  int flattened_input_dim = 1;
  for (int ii = 0; ii < in.dim() - 1; ++ii) {
    flattened_input_dim *= in.sizes()[ii];
  }
  ET_SWITCH_REAL_TYPES_AND2(
      Half, BFloat16, in.scalar_type(), ctx, "mm.out", CTYPE, [&]() {
        size_t n = flattened_input_dim;
        size_t k = in.sizes()[in.dim() - 1];
        size_t m = mat2.size(0);

        executorch::cpublas::gemm(
            executorch::cpublas::TransposeType::Transpose,
            executorch::cpublas::TransposeType::NoTranspose,
            m,
            n,
            k,
            static_cast<CTYPE>(1),
            mat2.const_data_ptr<CTYPE>(),
            k,
            in.const_data_ptr<CTYPE>(),
            k,
            static_cast<CTYPE>(0),
            out.mutable_data_ptr<CTYPE>(),
            m);
      });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
