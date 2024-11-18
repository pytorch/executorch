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

Tensor& opt_mm_out(
    RuntimeContext& ctx,
    const Tensor& in,
    const Tensor& mat2,
    Tensor& out) {
  ET_KERNEL_CHECK(ctx, check_mm_args(in, mat2, out), InvalidArgument, out);

  size_t output_ndim = 0;
  std::array<exec_aten::SizesType, kTensorDimensionLimit> output_sizes;
  get_mm_out_target_size(in, mat2, output_sizes.data(), &output_ndim);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {output_sizes.data(), output_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  if (out.numel() == 0) {
    return out;
  }
  ET_SWITCH_REAL_TYPES_AND2(
      Half, BFloat16, in.scalar_type(), ctx, "mm.out", CTYPE, [&]() {
        size_t n = in.size(0);
        size_t k = in.size(1);
        size_t m = mat2.size(1);

        // gemm expects column-major inputs and produces column-major
        // output. So, we take advantage of the identity (A @ B).t()
        // = B.t() @ A.t() here; row-major B is B.t() from gemm's
        // column-major perspective, etc.
        executorch::cpublas::gemm(
            executorch::cpublas::TransposeType::NoTranspose,
            executorch::cpublas::TransposeType::NoTranspose,
            m,
            n,
            k,
            static_cast<CTYPE>(1),
            mat2.const_data_ptr<CTYPE>(),
            m,
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
