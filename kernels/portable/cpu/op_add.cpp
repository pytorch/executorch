/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

Tensor& add_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    const Scalar& alpha,
    Tensor& out) {
  (void)ctx;

  resize_to_broadcast_target_size(a, b, out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = promoteTypes(a_type, b_type);
  ScalarType out_type = out.scalar_type();

  ET_CHECK(canCast(common_type, out_type));

  ET_SWITCH_REAL_TYPES_AND(Bool, a_type, ctx, "add", CTYPE_A, [&]() {
    ET_SWITCH_REAL_TYPES_AND(Bool, b_type, ctx, "add", CTYPE_B, [&]() {
      ET_SWITCH_REAL_TYPES_AND(Bool, common_type, ctx, "add", CTYPE_IN, [&]() {
        ET_SWITCH_REAL_TYPES_AND(Bool, out_type, ctx, "add", CTYPE_OUT, [&]() {
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
        });
      });
    });
  });

  return out;
}

Tensor& add_scalar_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    const Scalar& alpha,
    Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  auto error = resize_tensor(out, a.sizes());
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = utils::get_scalar_dtype(b);
  ScalarType common_type = utils::promote_type_with_scalar(a_type, b);
  ScalarType out_type = out.scalar_type();

  ET_CHECK(common_type == out_type);

  ET_SWITCH_REAL_TYPES_AND(Bool, a_type, ctx, "add", CTYPE_A, [&]() {
    ET_SWITCH_SCALAR_OBJ_TYPES(b_type, ctx, "add", CTYPE_B, [&]() {
      ET_SWITCH_REAL_TYPES_AND(Bool, common_type, ctx, "add", CTYPE_IN, [&]() {
        ET_SWITCH_REAL_TYPES_AND(Bool, out_type, ctx, "add", CTYPE_OUT, [&]() {
          CTYPE_B b_val;
          ET_EXTRACT_SCALAR(b, b_val);
          CTYPE_IN b_casted = static_cast<CTYPE_IN>(b_val);
          CTYPE_IN alpha_val;
          ET_EXTRACT_SCALAR(alpha, alpha_val);

          apply_unary_map_fn(
              [b_casted, alpha_val](const CTYPE_A val_a) {
                CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
                CTYPE_IN value = a_casted + alpha_val * b_casted;
                return static_cast<CTYPE_OUT>(value);
              },
              a.const_data_ptr<CTYPE_A>(),
              out.mutable_data_ptr<CTYPE_OUT>(),
              out.numel());
        });
      });
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
