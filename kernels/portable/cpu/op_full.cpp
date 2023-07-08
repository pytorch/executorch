// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

namespace {

/**
 * Fills the `out` with value of `value`.
 */
template <class CTYPE>
void full_kernel(const Scalar& value, Tensor& out) {
  CTYPE value_v;
  bool ok = utils::extract_scalar(value, &value_v);
  ET_CHECK_MSG(ok, "Invalid fill value: wrong type or out of range");
  const size_t n = out.numel();
  auto data_out = out.data_ptr<CTYPE>();
  for (size_t i = 0; i < n; ++i) {
    data_out[i] = value_v;
  }
}

inline void full_switch_out(
    const IntArrayRef sizes,
    const Scalar& fill_value,
    Tensor& out) {
#define FULL_IMPL_SWITCH_OUT_CASE(ctype, dtype) \
  case ScalarType::dtype:                       \
    full_kernel<ctype>(fill_value, out);        \
    break;

  switch (out.scalar_type()) {
    ET_FORALL_REAL_TYPES_AND(Bool, FULL_IMPL_SWITCH_OUT_CASE)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd", out.scalar_type());
  }

#undef FULL_IMPL_SWITCH_OUT_CASE
}

} // namespace

/**
 * Returns a tensor with a size filled with fill_value
 */
Tensor& full_out(
    RuntimeContext& context,
    const IntArrayRef sizes,
    const Scalar& fill_value,
    Tensor& out) {
  (void)context;

  Error err = resize_tensor(out, sizes);
  ET_CHECK_MSG(err == Error::Ok, "Could not resize out");

  full_switch_out(sizes, fill_value, out);

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
