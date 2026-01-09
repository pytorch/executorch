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
bool isnan_float(float x) {
  return std::isnan(x);
}

bool isnan_double(double x) {
  return std::isnan(x);
}

Tensor& isnan_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out) {
  return internal::unary_ufunc_realhbbf16_to_bool(
      isnan_float, isnan_double, ctx, in, out);
}

} // namespace native
} // namespace executor
} // namespace torch
