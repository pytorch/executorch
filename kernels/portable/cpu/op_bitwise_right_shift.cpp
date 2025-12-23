/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/pattern/bitwise_op.h>

namespace torch {
namespace executor {
namespace native {

Tensor& bitwise_right_shift_Tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "bitwise_right_shift.Tensor_out";
  return internal::bitwise_tensor_out<internal::bit_rshift, op_name>(
      ctx, a, b, out);
}

Tensor& bitwise_right_shift_Tensor_Scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] =
      "bitwise_right_shift.Tensor_Scalar_out";
  return internal::bitwise_scalar_out<internal::bit_rshift, op_name>(
      ctx, a, b, out);
}

} // namespace native
} // namespace executor
} // namespace torch
