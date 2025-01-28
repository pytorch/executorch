/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/matmul_ops_util.h>
#include <executorch/kernels/portable/cpu/vec_ops.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

Tensor& mm_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const Tensor& mat2,
    Tensor& out) {
  ET_KERNEL_CHECK(ctx, check_mm_args(in, mat2, out), InvalidArgument, out);

  size_t output_ndim = 0;
  exec_aten::SizesType output_sizes[kTensorDimensionLimit];
  get_mm_out_target_size(in, mat2, output_sizes, &output_ndim);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {output_sizes, output_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, mat2, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  ET_SWITCH_REAL_TYPES_AND2(
      Half, BFloat16, in.scalar_type(), ctx, "mm.out", CTYPE, [&]() {
        size_t m = in.size(0);
        size_t n = in.size(1);
        size_t p = mat2.size(1);

        vec_matmul<CTYPE>(
            out.mutable_data_ptr<CTYPE>(),
            in.const_data_ptr<CTYPE>(),
            mat2.const_data_ptr<CTYPE>(),
            m,
            n,
            p);
      });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
