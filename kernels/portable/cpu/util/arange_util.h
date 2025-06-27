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

executorch::aten::SizesType
compute_arange_out_size(double start, double end, double step);

inline executorch::aten::SizesType compute_arange_out_size(double end) {
  return compute_arange_out_size(0.0, end, 1.0);
}

void arange_out_impl(
    KernelRuntimeContext& ctx,
    double start,
    double end,
    double step,
    Tensor& out);

void arange_out_impl(KernelRuntimeContext& ctx, double end, Tensor& out);

inline void
arange_out_impl(double start, double end, double step, Tensor& out) {
  KernelRuntimeContext ctx;
  arange_out_impl(ctx, start, end, step, out);
}

inline void arange_out_impl(double end, Tensor& out) {
  KernelRuntimeContext ctx;
  arange_out_impl(ctx, 0.0, end, 1.0, out);
}
} // namespace torch::executor::native
