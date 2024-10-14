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

Tensor& bitwise_xor_Tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "bitwise_xor.Tensor_out";
  return internal::bitwise_tensor_out<op_name>(ctx, a, b, out);
}

Tensor& bitwise_xor_Scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "bitwise_xor.Scalar_out";
  return internal::bitwise_scalar_out<op_name>(ctx, a, b, out);
}

} // namespace native
} // namespace executor
} // namespace torch
