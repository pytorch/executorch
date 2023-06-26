// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cmath>

#include <executorch/kernels/kernel_includes.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

Tensor& logical_or_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  (void)ctx;

  // Determine output size and resize for dynamic shapes
  resize_to_broadcast_target_size(a, b, out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = promoteTypes(a_type, b_type);
  ScalarType out_type = out.scalar_type();

  ET_CHECK(canCast(common_type, out_type));

  ET_SWITCH_REAL_TYPES_AND(Bool, a_type, ctx, "logical_or", CTYPE_A, [&]() {
    ET_SWITCH_REAL_TYPES_AND(Bool, b_type, ctx, "logical_or", CTYPE_B, [&]() {
      ET_SWITCH_REAL_TYPES_AND(
          Bool, out_type, ctx, "logical_or", CTYPE_OUT, [&]() {
            apply_binary_elementwise_fn(
                [](const CTYPE_A val_a, const CTYPE_B val_b) {
                  bool a_casted = static_cast<bool>(val_a);
                  bool b_casted = static_cast<bool>(val_b);
                  bool value = a_casted || b_casted;

                  return static_cast<CTYPE_OUT>(value);
                },
                a,
                a.const_data_ptr<CTYPE_A>(),
                b,
                b.const_data_ptr<CTYPE_B>(),
                out,
                out.mutable_data_ptr<CTYPE_OUT>());
          });
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
