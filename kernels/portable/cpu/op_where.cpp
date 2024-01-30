/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

Tensor& where_out(
    RuntimeContext& ctx,
    const Tensor& cond,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  ScalarType cond_type = cond.scalar_type();
  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = promoteTypes(a_type, b_type);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, common_type == out_type, InvalidArgument, out);

  // Determine output size and resize for dynamic shapes
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, cond, out) == Error::Ok,
      InvalidArgument,
      out);

  ET_SWITCH_TWO_TYPES(
      Bool, Byte, cond_type, ctx, "where.self_out", CTYPE_COND, [&]() {
        ET_SWITCH_REAL_TYPES_AND(
            Bool, a_type, ctx, "where.self_out", CTYPE_A, [&]() {
              ET_SWITCH_REAL_TYPES_AND(
                  Bool, b_type, ctx, "where.self_out", CTYPE_B, [&]() {
                    ET_SWITCH_REAL_TYPES_AND(
                        Bool,
                        out_type,
                        ctx,
                        "where.self_out",
                        CTYPE_OUT,
                        [&]() {
                          apply_ternary_elementwise_fn<
                              CTYPE_A,
                              CTYPE_B,
                              CTYPE_COND,
                              CTYPE_OUT>(
                              [](const CTYPE_A val_a,
                                 const CTYPE_B val_b,
                                 const CTYPE_COND val_c) {
                                CTYPE_OUT a_casted =
                                    static_cast<CTYPE_OUT>(val_a);
                                CTYPE_OUT b_casted =
                                    static_cast<CTYPE_OUT>(val_b);
                                return val_c ? a_casted : b_casted;
                              },
                              a,
                              b,
                              cond,
                              out);
                        });
                  });
            });
      });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
