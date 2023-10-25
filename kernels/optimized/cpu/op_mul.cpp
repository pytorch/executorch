/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/optimized/vec/functional.h>
#include <executorch/kernels/optimized/vec/vec.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

Tensor& opt_mul_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  (void)ctx;

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType out_type = out.scalar_type();

  if (a_type == b_type && a_type == out_type && a.sizes().equals(b.sizes())) {
    // Resize for dynamic shape
    auto error = resize_tensor(out, a.sizes());
    ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");

    ET_SWITCH_REAL_TYPES_AND(Bool, out_type, ctx, "mul.out", CTYPE, [&]() {
      using Vec = executorch::vec::Vectorized<CTYPE>;
      executorch::vec::map2<CTYPE>(
          [](Vec x, Vec y) { return x * y; },
          out.mutable_data_ptr<CTYPE>(),
          a.const_data_ptr<CTYPE>(),
          b.const_data_ptr<CTYPE>(),
          out.numel());
    });
  } else {
    ScalarType common_type = promoteTypes(a_type, b_type);
    ET_CHECK(canCast(common_type, out_type));

    resize_to_broadcast_target_size(a, b, out);

    ET_SWITCH_REAL_TYPES_AND(Bool, a_type, ctx, "mul.out", CTYPE_A, [&]() {
      ET_SWITCH_REAL_TYPES_AND(Bool, b_type, ctx, "mul.out", CTYPE_B, [&]() {
        ET_SWITCH_REAL_TYPES_AND(
            Bool, common_type, ctx, "mul.out", CTYPE_IN, [&]() {
              ET_SWITCH_REAL_TYPES_AND(
                  Bool, out_type, ctx, "mul.out", CTYPE_OUT, [&]() {
                    apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
                        [](const CTYPE_A val_a, const CTYPE_B val_b) {
                          CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
                          CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
                          CTYPE_IN value = a_casted * b_casted;

                          return static_cast<CTYPE_OUT>(value);
                        },
                        a,
                        b,
                        out);
                  });
            });
      });
    });
  }

  return out;
}

Tensor& opt_mul_scalar_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  (void)ctx;

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = utils::get_scalar_dtype(b);
  ScalarType common_type = utils::promote_type_with_scalar(a_type, b);
  ScalarType out_type = out.scalar_type();

  ET_CHECK(common_type == out_type);

  // Resize for dynamic shape
  auto error = resize_tensor(out, a.sizes());
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");

  if (a_type == common_type && a_type == out_type) {
    ET_SWITCH_REAL_TYPES_AND(Bool, a_type, ctx, "mul.Scalar_out", CTYPE, [&]() {
      ET_SWITCH_REAL_TYPES_AND(
          Bool, b_type, ctx, "mul.Scalar_out", CTYPE_B, [&]() {
            CTYPE_B b_val;
            ET_EXTRACT_SCALAR(b, b_val);
            CTYPE b_casted = static_cast<CTYPE>(b_val);

            using Vec = executorch::vec::Vectorized<CTYPE>;
            executorch::vec::map<CTYPE>(
                [b_casted](Vec x) { return x * Vec(b_casted); },
                out.mutable_data_ptr<CTYPE>(),
                a.const_data_ptr<CTYPE>(),
                out.numel());
          });
    });
  } else {
    ET_SWITCH_REAL_TYPES_AND(
        Bool, a_type, ctx, "mul.Scalar_out", CTYPE_A, [&]() {
          ET_SWITCH_REAL_TYPES_AND(
              Bool, b_type, ctx, "mul.Scalar_out", CTYPE_B, [&]() {
                ET_SWITCH_REAL_TYPES_AND(
                    Bool, common_type, ctx, "mul.Scalar_out", CTYPE_IN, [&]() {
                      ET_SWITCH_REAL_TYPES_AND(
                          Bool,
                          out_type,
                          ctx,
                          "mul.Scalar_out",
                          CTYPE_OUT,
                          [&]() {
                            CTYPE_B b_val;
                            ET_EXTRACT_SCALAR(b, b_val);
                            CTYPE_IN b_casted = static_cast<CTYPE_IN>(b_val);

                            const size_t n = a.numel();
                            const CTYPE_A* a_data = a.const_data_ptr<CTYPE_A>();
                            CTYPE_OUT* out_data =
                                out.mutable_data_ptr<CTYPE_OUT>();
                            for (auto i = 0; i < n; ++i) {
                              out_data[i] = static_cast<CTYPE_OUT>(
                                  static_cast<CTYPE_IN>(a_data[i]) * b_casted);
                            }
                          });
                    });
              });
        });
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
