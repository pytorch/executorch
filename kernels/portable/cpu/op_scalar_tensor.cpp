// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

#include <executorch/kernels/portable/cpu/scalar_utils.h>

#include <cstdint>
#include <cstring>

namespace torch {
namespace executor {
namespace native {

using exec_aten::Tensor;

namespace {

template <typename CTYPE>
void set_tensor_value(const Scalar& s, Tensor& out) {
  CTYPE value = 0;
  bool ok = utils::extract_scalar(s, &value);
  ET_CHECK_MSG(ok, "Invalid other value: wrong type or out of range");

  CTYPE* out_data = out.data_ptr<CTYPE>();
  out_data[0] = value;
}

} // namespace

Tensor&
scalar_tensor_out(RuntimeContext& context, const Scalar& s, Tensor& out) {
  (void)context;

  ET_CHECK_MSG(out.numel() == 1, "Output tensor must have only one element");

#define SCALAR_TENSOR(ctype, dtype)  \
  case ScalarType::dtype:            \
    set_tensor_value<ctype>(s, out); \
    break;

  switch (out.scalar_type()) {
    ET_FORALL_REAL_TYPES(SCALAR_TENSOR);
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd", out.scalar_type());
  }
#undef SCALAR_TENSOR

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
