/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// patternlint-disable-next-line executorch-cpp-nostdinc
#include <functional>

#include <executorch/kernels/portable/cpu/pattern/comparison_op.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

Tensor& ge_tensor_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  return comparison_op_out<std::greater_equal>(ctx, a, b, out, "ge.Tensor_out");
}

Tensor& ge_scalar_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  return scalar_comparison_op_with_scalar_promotion_out<std::greater_equal>(
      ctx, a, b, out, "ge.Scalar_out");
}

} // namespace native
} // namespace executor
} // namespace torch
