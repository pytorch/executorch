/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/pattern/pattern.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {
namespace {

double rsqrt(double x) {
  return 1.0 / std::sqrt(x);
}

} // namespace

Tensor& rsqrt_out(RuntimeContext& ctx, const Tensor& in, Tensor& out) {
  return internal::unary_ufunc_realhb_to_floath(rsqrt, ctx, in, out);
}

} // namespace native
} // namespace executor
} // namespace torch
