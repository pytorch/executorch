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
// Passing std::isinf directly to unary_ufunc_realhb_to_bool can cause "error:
// cannot resolve overloaded function ‘isinf’ based on conversion to type
// ‘torch::executor::FunctionRef<bool(double)>’" in some compilation
// environments.
bool isinf_wrapper(double num) {
  return std::isinf(num);
}
} // namespace

Tensor& isinf_out(RuntimeContext& ctx, const Tensor& in, Tensor& out) {
  return internal::unary_ufunc_realhb_to_bool(isinf_wrapper, ctx, in, out);
}

} // namespace native
} // namespace executor
} // namespace torch
