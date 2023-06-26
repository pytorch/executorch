// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <executorch/kernels/kernel_includes.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;
using Scalar = exec_aten::Scalar;

namespace {

template <typename CTYPE_A, typename CTYPE_B, typename CTYPE_OUT>
void sub_tensors_impl(
    const Tensor& a,
    const Tensor& b,
    const Scalar& alpha,
    Tensor& out) {
  // Alpha multiplication is performed in double to maximize precision
  double alpha_val = 0;
  bool ok = utils::extract_scalar(alpha, &alpha_val);
  ET_CHECK_MSG(ok, "Invalid alpha value: wrong type or out of range");

  apply_binary_elementwise_fn(
      [alpha_val](const CTYPE_A val_a, const CTYPE_B val_b) {
        CTYPE_OUT a_casted = static_cast<CTYPE_OUT>(val_a);

        if (alpha_val == 1.0f) {
          CTYPE_OUT b_casted = static_cast<CTYPE_OUT>(val_b);
          return a_casted - b_casted;
        }

        double b_casted = static_cast<double>(val_b);
        return a_casted - static_cast<CTYPE_OUT>(alpha_val * b_casted);
      },
      a,
      a.data_ptr<CTYPE_A>(),
      b,
      b.data_ptr<CTYPE_B>(),
      out,
      out.data_ptr<CTYPE_OUT>());
}

template <typename CTYPE_A, typename CTYPE_B>
void sub_tensors_switch_out(
    const Tensor& a,
    const Tensor& b,
    const Scalar& alpha,
    Tensor& out) {
#define SUB_TENSORS_SWITCH_OUT_CASE(ctype, dtype)                \
  case ScalarType::dtype:                                        \
    sub_tensors_impl<CTYPE_A, CTYPE_B, ctype>(a, b, alpha, out); \
    break;

  switch (out.scalar_type()) {
    ET_FORALL_REAL_TYPES(SUB_TENSORS_SWITCH_OUT_CASE)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd for out", out.scalar_type());
  }

#undef SUB_TENSORS_SWITCH_OUT_CASE
}

template <typename CTYPE_A>
void sub_tensors_switch_b(
    const Tensor& a,
    const Tensor& b,
    const Scalar& alpha,
    Tensor& out) {
#define SUB_TENSORS_SWITCH_B_CASE(ctype, dtype)               \
  case ScalarType::dtype:                                     \
    sub_tensors_switch_out<CTYPE_A, ctype>(a, b, alpha, out); \
    break;

  switch (b.scalar_type()) {
    ET_FORALL_REAL_TYPES(SUB_TENSORS_SWITCH_B_CASE)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd for b", b.scalar_type());
  }

#undef SUB_TENSORS_SWITCH_B_CASE
}

void sub_tensors_switch_a(
    const Tensor& a,
    const Tensor& b,
    const Scalar& alpha,
    Tensor& out) {
#define SUB_TENSORS_SWITCH_A_CASE(ctype, dtype)    \
  case ScalarType::dtype:                          \
    sub_tensors_switch_b<ctype>(a, b, alpha, out); \
    break;

  switch (a.scalar_type()) {
    ET_FORALL_REAL_TYPES(SUB_TENSORS_SWITCH_A_CASE)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd for a", a.scalar_type());
  }

#undef SUB_TENSORS_SWITCH_A_CASE
}

void check_input_dtypes(
    const Tensor& a,
    const Tensor& b,
    const Scalar& alpha,
    Tensor& out) {
  // If either input is floating point, the output must also be floating point
  if (isFloatingType(a.scalar_type()) || isFloatingType(b.scalar_type())) {
    ET_CHECK_MSG(
        isFloatingType(out.scalar_type()),
        "output must be a floating point type if either input is a floating point type.");
  }
  // Bool output is only allowed if both inputs are bool
  if (out.scalar_type() == ScalarType::Bool) {
    ET_CHECK_MSG(
        a.scalar_type() == ScalarType::Bool &&
            b.scalar_type() == ScalarType::Bool,
        "both inputs must be bool type for output to be bool");
  }

  // If both inputs are integral or bool types, then alpha must also be an
  // integral type
  if (isIntegralType(a.scalar_type(), true) &&
      isIntegralType(b.scalar_type(), true)) {
    ET_CHECK_MSG(
        alpha.isIntegral(true),
        "alpha must be an integral type if both inputs are integral types");
  }
}

} // namespace

/**
 * Element-wise substraction of `b` from `a`, overwriting `out`.
 *
 * Asserts that all tensors have the same dtype and shape.
 *
 * sub.out(Tensor self, Tensor other, *, Scalar alpha=1.0, Tensor(a!) out) ->
 *     Tensor(a!)
 */
Tensor& sub_out(
    RuntimeContext& context,
    const Tensor& a,
    const Tensor& b,
    const Scalar& alpha,
    Tensor& out) {
  (void)context;

  // Determine output size and resize for dynamic shapes
  resize_to_broadcast_target_size(a, b, out);

  // Check arguments
  check_input_dtypes(a, b, alpha, out);

  sub_tensors_switch_a(a, b, alpha, out);

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
