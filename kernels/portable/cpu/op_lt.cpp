/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/pattern/comparison_op.h>

#include <functional>

namespace torch {
namespace executor {
namespace native {

Tensor& lt_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "lt.Tensor_out";
  return internal::comparison_tensor_out<std::less, op_name>(ctx, a, b, out);
}

Tensor& lt_scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "lt.Scalar_out";
  return internal::comparison_scalar_out<std::less, op_name>(ctx, a, b, out);
}

} // namespace native
} // namespace executor
} // namespace torch
