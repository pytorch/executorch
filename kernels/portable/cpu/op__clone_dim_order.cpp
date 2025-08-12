/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;

template <typename T>
using OptionalArrayRef = executorch::aten::OptionalArrayRef<T>;

/**
 * _clone_dim_order.out(Tensor self, *, bool non_blocking=False, int[]?
 * dim_order=None, Tensor(a!) out) -> Tensor(a!)
 *
 * Clones via element-wise copy while preserving dim_order.
 */
Tensor& _clone_dim_order_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    bool non_blocking,
    OptionalArrayRef<int64_t> dim_order,
    Tensor& out) {
  (void)ctx;

  // Ensure input and output dtype match.
  ET_KERNEL_CHECK(
      ctx, self.scalar_type() == out.scalar_type(), InvalidArgument, out);

  // Ensure output has the same layout as input or matches dim_order.
  ET_KERNEL_CHECK(
      ctx,
      check__to_dim_order_copy_args(self, non_blocking, dim_order, out),
      InvalidArgument,
      out);

  // Ensure input and output shapes match, resizing if necessary.
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, self.sizes()) == torch::executor::Error::Ok,
      InvalidArgument,
      out);

  if (self.numel() == 0) {
    return out;
  }

  // Select the correct input dtype and copy the tensors.
  ET_SWITCH_REALHBBF16_TYPES(
      self.scalar_type(),
      ctx,
      "dim_order_ops::_clone_dim_order.out",
      CTYPE,
      [&] { _to_dim_order_copy_impl<CTYPE, CTYPE>(self, out); });

  return out;
}

Tensor& _clone_dim_order_out(
    const Tensor& self,
    bool non_blocking,
    OptionalArrayRef<int64_t> dim_order,
    Tensor& out) {
  executorch::ET_RUNTIME_NAMESPACE::KernelRuntimeContext context{};
  return _clone_dim_order_out(context, self, non_blocking, dim_order, out);
}

} // namespace native
} // namespace executor
} // namespace torch