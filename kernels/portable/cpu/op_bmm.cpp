/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/matmul_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;

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
  executorch::aten::SizesType output_sizes[kTensorDimensionLimit];
  get_bmm_out_target_size(in, mat2, output_sizes, &output_ndim);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {output_sizes, output_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "bmm.out";

  auto in_type = in.scalar_type();

  if (executorch::runtime::isComplexType(in_type)) {
    ET_SWITCH_COMPLEXH_TYPES(in_type, ctx, op_name, CTYPE, [&]() {
      internal::bmm_out_impl<CTYPE>(in, mat2, out);
    });
  } else {
    ET_SWITCH_REALHBF16_TYPES(in_type, ctx, op_name, CTYPE, [&]() {
      internal::bmm_out_impl<CTYPE>(in, mat2, out);
    });
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
