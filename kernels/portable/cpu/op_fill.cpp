// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Scalar = exec_aten::Scalar;
using ScalarType = exec_aten::ScalarType;
using Tensor = exec_aten::Tensor;

Tensor& fill_scalar_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  (void)ctx;

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = utils::get_scalar_dtype(b);
  ScalarType out_type = out.scalar_type();

  ET_CHECK(a_type == out_type);

  // Resize for dynamic shape
  auto error = resize_tensor(out, a.sizes());
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");

  ET_SWITCH_REAL_TYPES_AND(Bool, a_type, ctx, "fill", CTYPE_A, [&] {
    CTYPE_A b_casted;
    ET_SWITCH_SCALAR_OBJ_TYPES(b_type, ctx, "fill", CTYPE_B, [&] {
      CTYPE_B b_val;
      ET_EXTRACT_SCALAR(b, b_val);
      b_casted = static_cast<CTYPE_A>(b_val);
    });

    apply_unary_map_fn(
        [b_casted](const CTYPE_A val_a) { return b_casted; },
        a.const_data_ptr<CTYPE_A>(),
        out.mutable_data_ptr<CTYPE_A>(),
        out.numel());
  });

  return out;
}

Tensor& fill_tensor_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  (void)ctx;

  // Assert `b` must be a scalar tensor.
  ET_CHECK(b.dim() == 0 && b.numel() == 1);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType out_type = out.scalar_type();

  ET_CHECK(a_type == out_type);

  // Resize for dynamic shape
  auto error = resize_tensor(out, a.sizes());
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");

  ET_SWITCH_REAL_TYPES_AND(Bool, a_type, ctx, "fill", CTYPE_A, [&] {
    CTYPE_A b_casted;
    ET_SWITCH_REAL_TYPES_AND(Bool, b_type, ctx, "fill", CTYPE_B, [&] {
      CTYPE_B b_val;
      ET_EXTRACT_SCALAR_TENSOR(b, b_val);
      b_casted = static_cast<CTYPE_A>(b_val);
    });

    apply_unary_map_fn(
        [b_casted](const CTYPE_A val_a) { return b_casted; },
        a.const_data_ptr<CTYPE_A>(),
        out.mutable_data_ptr<CTYPE_A>(),
        out.numel());
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
