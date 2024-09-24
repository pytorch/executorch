/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/padding_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

Tensor& reflection_pad2d_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    exec_aten::ArrayRef<int64_t> padding,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_padding_args(2, in, padding, out, /*reflection*/ true),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  Tensor::SizesType target_sizes[kTensorDimensionLimit];
  size_t target_ndim = 0;
  get_padding_out_target_size(2, in, padding, target_sizes, &target_ndim);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {target_sizes, target_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType in_type = in.scalar_type();
  constexpr auto name = "reflection_pad2d.out";

  ET_SWITCH_ALL_TYPES(in_type, ctx, name, CTYPE, [&] {
    pad2d<CTYPE>(reflection_ix, in, out, padding);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
