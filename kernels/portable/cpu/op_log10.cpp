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

Tensor& log10_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out) {
  static constexpr const char op_name[] = "log10.out";
  return internal::unary_ufunc_realhbbf16_to_floathbf16<op_name>(
      [](auto x) { return std::log10(x); }, ctx, in, out);
}

} // namespace native
} // namespace executor
} // namespace torch
