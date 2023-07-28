/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

Tensor& rsub_scalar_out(
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

  ET_SWITCH_REAL_TYPES(a_type, ctx, "rsub", CTYPE_A, [&]() {
    ET_SWITCH_SCALAR_OBJ_REAL_TYPES(b_type, ctx, "rsub", CTYPE_B, [&]() {
      ET_SWITCH_REAL_TYPES(common_type, ctx, "rsub", CTYPE_IN, [&]() {
        ET_SWITCH_REAL_TYPES(out_type, ctx, "rsub", CTYPE_OUT, [&]() {
          CTYPE_B b_val;
          ET_EXTRACT_SCALAR(b, b_val);
          CTYPE_IN b_casted = static_cast<CTYPE_IN>(b_val);
          CTYPE_IN alpha_val;
          ET_EXTRACT_SCALAR(alpha, alpha_val);

          apply_unary_map_fn(
              [b_casted, alpha_val](const CTYPE_A val_a) {
                CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
                CTYPE_IN value = b_casted - alpha_val * a_casted;
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
