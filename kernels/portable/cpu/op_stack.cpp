/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <executorch/kernels/portable/cpu/util/stack_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;

Tensor& stack_out(
    KernelRuntimeContext& ctx,
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  (void)ctx;

  if (dim < 0) {
    dim += out.dim();
  }

  ET_KERNEL_CHECK(
      ctx, check_stack_args(tensors, dim, out), InvalidArgument, out);

  for (size_t i = 0; i < tensors.size(); ++i) {
    ET_KERNEL_CHECK(
        ctx,
        tensors_have_same_dim_order(tensors[i], out),
        InvalidArgument,
        out);
  }

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(out), InvalidArgument, out);

  Tensor::SizesType expected_out_size[kTensorDimensionLimit];
  size_t expected_out_dim = 0;
  get_stack_out_target_size(tensors, dim, expected_out_size, &expected_out_dim);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {expected_out_size, expected_out_dim}) == Error::Ok,
      InvalidArgument,
      out);
  return stack_out_impl(ctx, tensors, dim, out);
}

} // namespace native
} // namespace executor
} // namespace torch
