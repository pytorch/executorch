// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <executorch/kernels/kernel_includes.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

namespace {

ScalarType get_compute_type(ScalarType a_type, ScalarType b_type) {
  ET_CHECK(
      !isComplexType(a_type) && !isQIntType(a_type) && !isBitsType(a_type));
  ET_CHECK(
      !isComplexType(b_type) && !isQIntType(b_type) && !isBitsType(b_type));

  if (isFloatingType(a_type) && isFloatingType(b_type)) {
    return promoteTypes(a_type, b_type);
  } else if (isFloatingType(a_type)) {
    return a_type;
  } else if (isFloatingType(b_type)) {
    return b_type;
  }
  return ScalarType::Float;
}

} // namespace

Tensor&
div_out(RuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out) {
  (void)ctx;

  resize_to_broadcast_target_size(a, b, out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = get_compute_type(a_type, b_type);
  ScalarType out_type = out.scalar_type();

  ET_CHECK(canCast(common_type, out_type));

  ET_SWITCH_REAL_TYPES_AND(Bool, a_type, ctx, "div", CTYPE_A, [&]() {
    ET_SWITCH_REAL_TYPES_AND(Bool, b_type, ctx, "div", CTYPE_B, [&]() {
      ET_SWITCH_FLOAT_TYPES(common_type, ctx, "div", CTYPE_IN, [&]() {
        ET_SWITCH_FLOAT_TYPES(out_type, ctx, "div", CTYPE_OUT, [&]() {
          apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
              [](const CTYPE_A val_a, const CTYPE_B val_b) {
                CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
                CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
                CTYPE_IN value = a_casted / b_casted;

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

Tensor& div_scalar_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  auto error = resize_tensor(out, a.sizes());
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = utils::get_scalar_dtype(b);
  ScalarType common_type = isFloatingType(a_type) ? a_type : ScalarType::Float;
  ScalarType out_type = out.scalar_type();

  ET_CHECK(common_type == out_type);

  ET_SWITCH_REAL_TYPES_AND(Bool, a_type, ctx, "div", CTYPE_A, [&]() {
    ET_SWITCH_REAL_TYPES_AND(Bool, b_type, ctx, "div", CTYPE_B, [&]() {
      ET_SWITCH_REAL_TYPES(common_type, ctx, "div", CTYPE_IN, [&]() {
        ET_SWITCH_REAL_TYPES(out_type, ctx, "div", CTYPE_OUT, [&]() {
          CTYPE_B b_val;
          ET_EXTRACT_SCALAR(b, b_val);
          CTYPE_IN b_casted = static_cast<CTYPE_IN>(b_val);

          apply_unary_map_fn(
              [b_casted](const CTYPE_A val_a) {
                CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
                CTYPE_IN value = a_casted / b_casted;
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
