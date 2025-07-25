/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

Tensor& add_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    const Scalar& alpha,
    Tensor& out) {
  (void)ctx;

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = promoteTypes(a_type, b_type);
  ScalarType out_type = out.scalar_type();

  ET_CHECK_MSG(a_type == ScalarType::Float, "Input tensor not a float.\n");
  ET_CHECK_MSG(b_type == ScalarType::Float, "Input tensor not a float.\n");
  ET_CHECK_MSG(out_type == ScalarType::Float, "Output tensor not a float.\n");

  ET_CHECK(canCast(common_type, out_type));

  using CTYPE_A = float;
  using CTYPE_B = float;
  using CTYPE_IN = float;
  using CTYPE_OUT = float;
  CTYPE_IN alpha_val;
  ET_EXTRACT_SCALAR(alpha, alpha_val);

  apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
      [alpha_val](const CTYPE_A val_a, const CTYPE_B val_b) {
        CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
        CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
        CTYPE_IN value = a_casted + alpha_val * b_casted;

        return static_cast<CTYPE_OUT>(value);
      },
      a,
      b,
      out);

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
