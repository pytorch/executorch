// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <executorch/core/Assert.h>
#include <executorch/kernels/kernel_includes.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

namespace {

template <typename CTYPE_A, typename CTYPE_B, typename CTYPE_OUT>
void mul_tensors_impl(const Tensor& a, const Tensor& b, Tensor& out) {
  apply_binary_elementwise_fn(
      [](const CTYPE_A val_a, const CTYPE_B val_b) {
        // Perform math in double for all types to maximize precision
        double a_casted = static_cast<double>(val_a);
        double b_casted = static_cast<double>(val_b);
        double value = a_casted * b_casted;

        return static_cast<CTYPE_OUT>(value);
      },
      a,
      a.data_ptr<CTYPE_A>(),
      b,
      b.data_ptr<CTYPE_B>(),
      out,
      out.data_ptr<CTYPE_OUT>());
}

template <typename CTYPE_A, typename CTYPE_B>
void mul_tensors_switch_out(const Tensor& a, const Tensor& b, Tensor& out) {
#define MUL_TENSORS_SWITCH_OUT_CASE(ctype, dtype)         \
  case ScalarType::dtype:                                 \
    mul_tensors_impl<CTYPE_A, CTYPE_B, ctype>(a, b, out); \
    break;

  switch (out.scalar_type()) {
    ET_FORALL_REAL_TYPES_AND(Bool, MUL_TENSORS_SWITCH_OUT_CASE)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd for out", out.scalar_type());
  }

#undef MUL_TENSORS_SWITCH_OUT_CASE
}

template <typename CTYPE_A>
void mul_tensors_switch_b(const Tensor& a, const Tensor& b, Tensor& out) {
#define MUL_TENSORS_SWITCH_B_CASE(ctype, dtype)        \
  case ScalarType::dtype:                              \
    mul_tensors_switch_out<CTYPE_A, ctype>(a, b, out); \
    break;

  switch (b.scalar_type()) {
    ET_FORALL_REAL_TYPES_AND(Bool, MUL_TENSORS_SWITCH_B_CASE)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd for b", b.scalar_type());
  }

#undef MUL_TENSORS_SWITCH_B_CASE
}

void mul_tensors_switch_a(const Tensor& a, const Tensor& b, Tensor& out) {
#define MUL_TENSORS_SWITCH_A_CASE(ctype, dtype) \
  case ScalarType::dtype:                       \
    mul_tensors_switch_b<ctype>(a, b, out);     \
    break;

  switch (a.scalar_type()) {
    ET_FORALL_REAL_TYPES_AND(Bool, MUL_TENSORS_SWITCH_A_CASE)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd for a", a.scalar_type());
  }

#undef MUL_TENSORS_SWITCH_A_CASE
}

void check_input_dtypes(const Tensor& a, const Tensor& b, Tensor& out) {
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
}

} // namespace

Tensor& mul_out(
    RuntimeContext& context,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  (void)context;

  // Determine output size and resize for dynamic shapes
  resize_to_broadcast_target_size(a, b, out);

  // Check arguments
  check_input_dtypes(a, b, out);

  mul_tensors_switch_a(a, b, out);

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
