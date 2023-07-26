// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <executorch/kernels/portable/cpu/pattern/pattern.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/core/function_ref.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {
namespace internal {

Tensor& binary_ufunc_realb_realb_to_realb_logical(
    FunctionRef<bool(bool, bool)> fn,
    RuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  (void)ctx;

  // Determine output size and resize for dynamic shapes
  resize_to_broadcast_target_size(a, b, out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType out_type = out.scalar_type();

  ET_SWITCH_REAL_TYPES_AND(Bool, a_type, ctx, __func__, CTYPE_A, [&]() {
    ET_SWITCH_REAL_TYPES_AND(Bool, b_type, ctx, __func__, CTYPE_B, [&]() {
      ET_SWITCH_REAL_TYPES_AND(Bool, out_type, ctx, __func__, CTYPE_OUT, [&]() {
        apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
            [fn](const CTYPE_A val_a, const CTYPE_B val_b) {
              bool a_casted = static_cast<bool>(val_a);
              bool b_casted = static_cast<bool>(val_b);
              bool value = fn(a_casted, b_casted);

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

} // namespace internal
} // namespace native
} // namespace executor
} // namespace torch
