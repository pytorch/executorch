/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/pattern/pattern.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <cmath>

namespace torch {
namespace executor {
namespace native {
namespace {

bool logical_or(bool a, bool b) {
  return a || b;
}

} // namespace

Tensor& logical_or_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  return internal::binary_ufunc_realb_realb_to_realb_logical(
      logical_or, ctx, a, b, out);
}

} // namespace native
} // namespace executor
} // namespace torch
