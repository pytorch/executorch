/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch::executor::native {

bool check_stack_args(
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out);

void get_stack_out_target_size(
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    executorch::aten::SizesType* out_sizes,
    size_t* out_ndim);

Tensor& stack_out_impl(
    KernelRuntimeContext& ctx,
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out);

inline Tensor& stack_out_impl(
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  KernelRuntimeContext ctx;
  return stack_out_impl(ctx, tensors, dim, out);
}

} // namespace torch::executor::native
