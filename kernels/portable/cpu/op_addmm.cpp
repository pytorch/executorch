/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/matmul_ops_util.h>
#include <executorch/kernels/portable/cpu/vec_ops.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using Scalar = executorch::aten::Scalar;

Tensor& addmm_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      check_addmm_args(in, mat1, mat2, beta, alpha, out),
      InvalidArgument,
      out);

  size_t output_ndim = 0;
  executorch::aten::SizesType output_sizes[kTensorDimensionLimit];
  get_mm_out_target_size(mat1, mat2, output_sizes, &output_ndim);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {output_sizes, output_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensor_is_broadcastable_to(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx,
      tensors_have_same_dim_order(in, mat1, mat2, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  ET_DEFINE_OPERATOR_NAME(op_name, "addmm.out");

  // in broadcasts to [m, p] and has at most 2 dims. A missing or size-1 dim
  // gets stride 0, so every bias shape takes the same single-rounding path.
  const ssize_t in_dim = in.dim();
  const int64_t col_stride = in_dim >= 1 && in.size(in_dim - 1) != 1 ? 1 : 0;
  const int64_t row_stride = in_dim == 2 && in.size(0) != 1 ? in.size(1) : 0;

  ET_SWITCH_REALHBF16_TYPES(in.scalar_type(), ctx, op_name, CTYPE, [&]() {
    using ACC = typename executorch::runtime::
        promote_types<CTYPE, CTYPE, /*half_to_float=*/true>::type;
    vec_addmm<CTYPE>(
        out.mutable_data_ptr<CTYPE>(),
        in.const_data_ptr<CTYPE>(),
        mat1.const_data_ptr<CTYPE>(),
        mat2.const_data_ptr<CTYPE>(),
        mat1.size(0),
        mat1.size(1),
        mat2.size(1),
        utils::scalar_to<ACC>(beta),
        utils::scalar_to<ACC>(alpha),
        row_stride,
        col_stride);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
