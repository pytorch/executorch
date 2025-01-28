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

Tensor& bmm_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const Tensor& mat2,
    Tensor& out) {
  ET_KERNEL_CHECK(ctx, check_bmm_args(in, mat2, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, mat2, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  size_t output_ndim = 0;
  exec_aten::SizesType output_sizes[kTensorDimensionLimit];
  get_bmm_out_target_size(in, mat2, output_sizes, &output_ndim);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {output_sizes, output_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  ET_SWITCH_REAL_TYPES_AND(
      Half, in.scalar_type(), ctx, "bmm.out", CTYPE, [&]() {
        const CTYPE* in_data = in.const_data_ptr<CTYPE>();
        const CTYPE* mat2_data = mat2.const_data_ptr<CTYPE>();
        CTYPE* out_data = out.mutable_data_ptr<CTYPE>();

        int64_t batch_size = in.size(0);
        int64_t m = in.size(1);
        int64_t n = in.size(2);
        int64_t p = mat2.size(2);

        for (int i = 0; i < batch_size; ++i) {
          const CTYPE* in_data_offset = in_data + i * m * n;
          const CTYPE* mat2_data_offset = mat2_data + i * n * p;
          CTYPE* out_data_offset = out_data + i * m * p;

          vec_matmul<CTYPE>(
              out_data_offset, in_data_offset, mat2_data_offset, m, n, p);
        }
      });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
