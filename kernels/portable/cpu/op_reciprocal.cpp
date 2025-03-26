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

float reciprocal(float x) {
  return 1.0f / x;
}

double reciprocal(double x) {
  return 1.0 / x;
}

template <
    typename Integer,
    std::enable_if_t<std::is_integral_v<Integer>, bool> = true>
double reciprocal(Integer x) {
  return reciprocal((double)x);
}
} // namespace

Tensor&
reciprocal_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out) {
  static constexpr const char op_name[] = "reciprocal.out";
  return internal::unary_ufunc_realhbbf16_to_floathbf16<op_name>(
      [](auto x) { return reciprocal(x); }, ctx, in, out);
}

} // namespace native
} // namespace executor
} // namespace torch
