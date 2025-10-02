/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/stack_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

Tensor& stack_out(
    KernelRuntimeContext& ctx,
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  return utils::stack_out_impl(ctx, tensors, dim, out);
}

} // namespace native
} // namespace executor
} // namespace torch
