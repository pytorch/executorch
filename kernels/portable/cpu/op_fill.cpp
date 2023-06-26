// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <executorch/kernels/kernel_includes.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>

namespace torch {
namespace executor {
namespace native {

using Scalar = exec_aten::Scalar;
using ScalarType = exec_aten::ScalarType;

namespace {

template <class CTYPE>
void fill_scalar_kernel(const Tensor& self, const Scalar& other, Tensor& out) {
  // Assert `self` and `out` have the same number of elements.
  ET_CHECK(self.numel() == out.numel());

  // Assert `other` is a valid scalar.
  CTYPE other_v = 0;
  bool ok = utils::extract_scalar(other, &other_v);
  ET_CHECK_MSG(ok, "Invalid other value: wrong type or out of range");

  // Create pointer over `out` data with `CTYPE`.
  auto data_out = out.data_ptr<CTYPE>();

  // Set each element of `out` data to `other_v`.
  for (auto i = 0; i < out.numel(); i++) {
    data_out[i] = static_cast<CTYPE>(other_v);
  }
}

template <class CTYPE>
void fill_tensor_kernel(const Tensor& self, const Tensor& other, Tensor& out) {
  // Assert `self` and `out` have the same number of elements.
  ET_CHECK(self.numel() == out.numel());

  // Assert `other` is a valid scalar.
  CTYPE other_v = 0;
  bool ok = extract_scalar_tensor(other, &other_v);
  ET_CHECK_MSG(ok, "Invalid other value: wrong type or out of range");

  // Create pointer over `other` and `out` data with `CTYPE`.
  auto data_out = out.data_ptr<CTYPE>();

  // Set each element of `out` data to `other_v`.
  for (auto i = 0; i < out.numel(); i++) {
    data_out[i] = static_cast<CTYPE>(other_v);
  }
}

} // namespace

Tensor& fill_scalar_out(
    RuntimeContext& context,
    const Tensor& self,
    const Scalar& other,
    Tensor& out) {
  (void)context;

  // Assert `self` and `out` have the same tensor shape.
  ET_CHECK_SAME_SHAPE2(self, out);

#define FILL_SCALAR_OUT(ctype, dtype)            \
  case ScalarType::dtype:                        \
    fill_scalar_kernel<ctype>(self, other, out); \
    break;

  switch (self.scalar_type()) {
    ET_FORALL_REAL_TYPES_AND(Bool, FILL_SCALAR_OUT)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd", self.scalar_type());
  }

#undef FILL_SCALAR_OUT

  return out;
}

Tensor& fill_tensor_out(
    RuntimeContext& context,
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  (void)context;

  // Assert `other` must be a scalar tensor.
  ET_CHECK(other.dim() == 0 && other.numel() == 1);

  // Assert `self`, `other`, and `out` have the same dtype.
  ET_CHECK_SAME_DTYPE3(self, other, out);

#define FILL_TENSOR_OUT(ctype, dtype)            \
  case ScalarType::dtype:                        \
    fill_tensor_kernel<ctype>(self, other, out); \
    break;

  switch (self.scalar_type()) {
    ET_FORALL_REAL_TYPES_AND(Bool, FILL_TENSOR_OUT)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd", self.scalar_type());
  }

#undef FILL_TENSOR_OUT

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
