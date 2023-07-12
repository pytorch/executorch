// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;
using Scalar = exec_aten::Scalar;
namespace {

/**
 * Substracts `a`, scaled by `alpha` from scalar `b`, overwriting `out`.
 *
 * Assumes that the tensors are contiguous, are the same shape, and have the
 * same dtype. CTYPE should be the C type (like `float` or `int`) that matches
 * the dtype of the tensors.
 */
template <class CTYPE>
void rsub_tensor_scalar(
    const Tensor& a,
    const Scalar& b,
    const Scalar& alpha,
    Tensor& out) {
  ET_DCHECK(a.numel() == out.numel());
  CTYPE alpha_v = 0;
  CTYPE b_v = 0;
  bool ok = utils::extract_scalar(alpha, &alpha_v);
  ET_CHECK_MSG(ok, "Invalid alpha value: wrong type or out of range");
  ok = utils::extract_scalar(b, &b_v);
  ET_CHECK_MSG(ok, "Invalid b value: wrong type or out of range");

  const size_t n = a.numel();
  const auto data_a = a.data_ptr<CTYPE>();
  auto data_out = out.data_ptr<CTYPE>();
  for (size_t i = 0; i < n; ++i) {
    data_out[i] = b_v - alpha_v * data_a[i];
  }
}

} // namespace

/**
 * Element-wise substraction of `a` from scalar `b`, overwriting `out`.
 *
 * Asserts that all tensors have the same dtype and shape.
 *
 * rsub.Scalar.out(Tensor self, Scalar other, *, Scalar alpha=1.0, Tensor(a!)
 * out) -> Tensor(a!)
 */
Tensor& rsub_scalar_out(
    const Tensor& a,
    const Scalar& b,
    const Scalar& alpha,
    Tensor& out) {
  ET_CHECK_SAME_SHAPE_AND_DTYPE2(a, out);

#define SUB_TENSORS(ctype, dtype)                \
  case ScalarType::dtype:                        \
    rsub_tensor_scalar<ctype>(a, b, alpha, out); \
    break;

  switch (a.scalar_type()) {
    ET_FORALL_REAL_TYPES(SUB_TENSORS)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd", a.scalar_type());
  }

#undef SUB_TENSORS

  return out;
}

Tensor& rsub_scalar_out(
    RuntimeContext& context,
    const Tensor& a,
    const Scalar& b,
    const Scalar& alpha,
    Tensor& out) {
  // TODO(larryliu): Add a context arg to the real op function and remove this
  // wrapper
  (void)context;
  return rsub_scalar_out(a, b, alpha, out);
}
} // namespace native
} // namespace executor
} // namespace torch
