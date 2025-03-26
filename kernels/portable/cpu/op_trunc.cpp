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

Tensor& trunc_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out) {
  static constexpr const char op_name[] = "trunc.out";
  return internal::unary_ufunc_realh<op_name>(
      [](auto x) { return std::trunc(x); }, ctx, in, out);
}

} // namespace native
} // namespace executor
} // namespace torch
