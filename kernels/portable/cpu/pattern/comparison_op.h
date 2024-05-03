/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {
template <template <typename> typename OpFunc>
Tensor& comparison_op_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out,
    const char* op_name) {
  /* Determine output size and resize for dynamic shapes */
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType out_type = out.scalar_type();

  ET_SWITCH_REAL_TYPES_AND(Bool, a_type, ctx, op_name, CTYPE_A, [&]() {
    ET_SWITCH_REAL_TYPES_AND(Bool, b_type, ctx, op_name, CTYPE_B, [&]() {
      using CTYPE_IN =
          typename torch::executor::promote_types<CTYPE_A, CTYPE_B>::type;
      ET_DCHECK(
          CppTypeToScalarType<CTYPE_IN>::value == promoteTypes(a_type, b_type));
      ET_SWITCH_REAL_TYPES_AND(Bool, out_type, ctx, op_name, CTYPE_OUT, [&]() {
        apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
            [](const CTYPE_A val_a, const CTYPE_B val_b) {
              CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
              CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
              bool value = OpFunc<CTYPE_IN>()(a_casted, b_casted);
              return static_cast<CTYPE_OUT>(value);
            },
            a,
            b,
            out);
      });
    });
  });
  return out;
}
} // namespace native
} // namespace executor
} // namespace torch
