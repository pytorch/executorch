/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/pattern/logical_op.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <cmath>

namespace torch {
namespace executor {
namespace native {
namespace {

bool logical_and(bool a, bool b) {
  return a && b;
}

} // namespace

Tensor& logical_and_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "logical_and.out";
  return internal::logical_tensor_out<op_name>(logical_and, ctx, a, b, out);
}

} // namespace native
} // namespace executor
} // namespace torch
