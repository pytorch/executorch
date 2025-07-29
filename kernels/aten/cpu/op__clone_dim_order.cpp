/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/aten/cpu/util/copy_ops_util.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using MemoryFormat = executorch::aten::MemoryFormat;

template <typename T>
using OptionalArrayRef = executorch::aten::OptionalArrayRef<T>;

template <typename T>
using Optional = std::optional<T>;

/**
 * _clone_dim_order.out(Tensor self, *, bool non_blocking=False, int[]?
 * dim_order=None, Tensor(a!) out) -> Tensor(a!)
 *
 * Clones with explicit dim_order, using the corresponding memory format.
 */
Tensor& _clone_dim_order_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    bool non_blocking,
    OptionalArrayRef<int64_t> dim_order,
    Tensor& out) {
  // Ensure output has the same layout as input or matches dim_order.
  ET_KERNEL_CHECK(
      ctx,
      check__to_dim_order_copy_args(self, non_blocking, dim_order, out),
      InvalidArgument,
      out);

  Optional<MemoryFormat> memory_format = get_memory_format(dim_order);
  at::clone_outf(self, memory_format, out);

  return out;
}

Tensor& _clone_dim_order_out(
    const Tensor& self,
    bool non_blocking,
    OptionalArrayRef<int64_t> dim_order,
    Tensor& out) {
  KernelRuntimeContext ctx{};
  return _clone_dim_order_out(ctx, self, non_blocking, dim_order, out);
}

} // namespace native
} // namespace executor
} // namespace torch