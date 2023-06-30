// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <executorch/core/Assert.h>
#include <executorch/kernels/kernel_includes.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>

namespace torch {
namespace executor {
namespace native {

Tensor& sub_out(
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

  ET_SWITCH_REAL_TYPES(a_type, ctx, "sub", CTYPE_A, [&]() {
    ET_SWITCH_REAL_TYPES(b_type, ctx, "sub", CTYPE_B, [&]() {
      ET_SWITCH_REAL_TYPES(common_type, ctx, "sub", CTYPE_IN, [&]() {
        ET_SWITCH_REAL_TYPES(out_type, ctx, "sub", CTYPE_OUT, [&]() {
          CTYPE_IN alpha_val;
          ET_EXTRACT_SCALAR(alpha, alpha_val);

          apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
              [alpha_val](const CTYPE_A val_a, const CTYPE_B val_b) {
                CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
                CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
                CTYPE_IN value = a_casted - alpha_val * b_casted;

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

Tensor& sub_scalar_out(
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

  ET_SWITCH_REAL_TYPES(a_type, ctx, "sub", CTYPE_A, [&]() {
    ET_SWITCH_REAL_TYPES(b_type, ctx, "sub", CTYPE_B, [&]() {
      ET_SWITCH_REAL_TYPES(common_type, ctx, "sub", CTYPE_IN, [&]() {
        ET_SWITCH_REAL_TYPES(out_type, ctx, "sub", CTYPE_OUT, [&]() {
          CTYPE_B b_val;
          ET_EXTRACT_SCALAR(b, b_val);
          CTYPE_IN b_casted = static_cast<CTYPE_IN>(b_val);
          CTYPE_IN alpha_val;
          ET_EXTRACT_SCALAR(alpha, alpha_val);

          apply_unary_map_fn(
              [b_casted, alpha_val](const CTYPE_A val_a) {
                CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
                CTYPE_IN value = a_casted - alpha_val * b_casted;
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
