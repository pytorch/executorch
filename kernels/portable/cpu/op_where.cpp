#include <executorch/kernels/kernel_includes.h>
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
void where_tensors_impl(
    const Tensor& condition,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  apply_ternary_elementwise_fn<CTYPE_A, CTYPE_B, bool, CTYPE_OUT>(
      [](const CTYPE_A val_a, const CTYPE_B val_b, const bool val_c) {
        CTYPE_OUT a_casted = static_cast<CTYPE_OUT>(val_a);
        CTYPE_OUT b_casted = static_cast<CTYPE_OUT>(val_b);

        return val_c ? a_casted : b_casted;
      },
      a,
      b,
      condition,
      out);
}

template <typename CTYPE_A, typename CTYPE_B>
void where_tensors_switch_out(
    const Tensor& condition,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
#define WHERE_TENSORS_SWITCH_OUT_CASE(ctype, dtype)                    \
  case ScalarType::dtype:                                              \
    where_tensors_impl<CTYPE_A, CTYPE_B, ctype>(condition, a, b, out); \
    break;

  switch (out.scalar_type()) {
    ET_FORALL_REAL_TYPES_AND(Bool, WHERE_TENSORS_SWITCH_OUT_CASE)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd for out", out.scalar_type());
  }

#undef WHERE_TENSORS_SWITCH_OUT_CASE
}

template <typename CTYPE_A>
void where_tensors_switch_b(
    const Tensor& condition,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
#define WHERE_TENSORS_SWITCH_B_CASE(ctype, dtype)                   \
  case ScalarType::dtype:                                           \
    where_tensors_switch_out<CTYPE_A, ctype>(condition, a, b, out); \
    break;

  switch (b.scalar_type()) {
    ET_FORALL_REAL_TYPES_AND(Bool, WHERE_TENSORS_SWITCH_B_CASE)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd for b", b.scalar_type());
  }

#undef WHERE_TENSORS_SWITCH_B_CASE
}

void where_tensors_switch_a(
    const Tensor& condition,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
#define WHERE_TENSORS_SWITCH_A_CASE(ctype, dtype)        \
  case ScalarType::dtype:                                \
    where_tensors_switch_b<ctype>(condition, a, b, out); \
    break;

  switch (a.scalar_type()) {
    ET_FORALL_REAL_TYPES_AND(Bool, WHERE_TENSORS_SWITCH_A_CASE)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd for a", a.scalar_type());
  }

#undef WHERE_TENSORS_SWITCH_A_CASE
}

void check_input_dtypes(
    const Tensor& condition,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  ET_CHECK_MSG(
      condition.scalar_type() == ScalarType::Bool,
      "Condition tensor must be boolean type");
}

} // namespace

Tensor& where_out(
    RuntimeContext& context,
    const Tensor& condition,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  (void)context;

  // Determine output size and resize for dynamic shapes
  resize_to_broadcast_target_size(a, b, condition, out);

  // Check arguments
  check_input_dtypes(condition, a, b, out);

  where_tensors_switch_a(condition, a, b, out);

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
