// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <executorch/core/Assert.h>
#include <executorch/kernels/kernel_includes.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <cmath>
#include <type_traits>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

namespace {

template <typename CTYPE_A, typename CTYPE_B, typename CTYPE_OUT>
void div_tensors_impl(const Tensor& a, const Tensor& b, Tensor& out) {
  apply_binary_elementwise_fn(
      [](const CTYPE_A val_a, const CTYPE_B val_b) {
        // Perform math in double for all types to maximize precision
        double dividend = static_cast<double>(val_a);
        double divisor = static_cast<double>(val_b);
        double value = dividend / divisor;

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
void div_tensors_switch_out(const Tensor& a, const Tensor& b, Tensor& out) {
#define DIV_TENSORS_SWITCH_OUT_CASE(ctype, dtype)         \
  case ScalarType::dtype:                                 \
    div_tensors_impl<CTYPE_A, CTYPE_B, ctype>(a, b, out); \
    break;

  switch (out.scalar_type()) {
    ET_FORALL_FLOAT_TYPES(DIV_TENSORS_SWITCH_OUT_CASE)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd for out", out.scalar_type());
  }

#undef DIV_TENSORS_SWITCH_OUT_CASE
}

template <typename CTYPE_A>
void div_tensors_switch_b(const Tensor& a, const Tensor& b, Tensor& out) {
#define DIV_TENSORS_SWITCH_B_CASE(ctype, dtype)        \
  case ScalarType::dtype:                              \
    div_tensors_switch_out<CTYPE_A, ctype>(a, b, out); \
    break;

  switch (b.scalar_type()) {
    ET_FORALL_REAL_TYPES_AND(Bool, DIV_TENSORS_SWITCH_B_CASE)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd for b", b.scalar_type());
  }

#undef DIV_TENSORS_SWITCH_B_CASE
}

void div_tensors_switch_a(const Tensor& a, const Tensor& b, Tensor& out) {
#define DIV_TENSORS_SWITCH_A_CASE(ctype, dtype) \
  case ScalarType::dtype:                       \
    div_tensors_switch_b<ctype>(a, b, out);     \
    break;

  switch (a.scalar_type()) {
    ET_FORALL_REAL_TYPES_AND(Bool, DIV_TENSORS_SWITCH_A_CASE)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd for a", a.scalar_type());
  }

#undef DIV_TENSORS_SWITCH_A_CASE
}

void check_input_dtypes(const Tensor& a, const Tensor& b, Tensor& out) {
  ET_CHECK_MSG(
      isFloatingType(out.scalar_type()),
      "output must be a floating point type.");
}

} // namespace

Tensor& div_out(
    RuntimeContext& context,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  (void)context;

  // Determine output size and resize for dynamic shapes
  resize_to_broadcast_target_size(a, b, out);

  // Check arguments
  check_input_dtypes(a, b, out);

  div_tensors_switch_a(a, b, out);

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
