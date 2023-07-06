// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <executorch/kernels/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

namespace {
template <class CTYPE>
void any_all_out(const Tensor& self, Tensor& out) {
  const size_t n = self.numel();
  const auto data_self = self.data_ptr<CTYPE>();

  auto data_out = out.data_ptr<bool>();
  data_out[0] = false;
  for (auto i = 0; i < n; ++i) {
    if (static_cast<bool>(data_self[i])) {
      data_out[0] = true;
      break;
    }
  }
}
} // namespace

// Tests if any element in input evaluates to True.
Tensor& any_all_out(RuntimeContext& context, const Tensor& self, Tensor& out) {
  (void)context;
  // Only support Boolean as output type.
  ET_CHECK_MSG(
      out.scalar_type() == ScalarType::Bool,
      "dtype of the output Tensor shall be Boolean.");
  ET_CHECK_MSG(out.dim() == 0, "dimension of the output Tensor shall be 0.");

#define ANY_ALL_OUT(ctype, dtype)  \
  case ScalarType::dtype:          \
    any_all_out<ctype>(self, out); \
    break;

  switch (self.scalar_type()) {
    ET_FORALL_REAL_TYPES_AND(Bool, ANY_ALL_OUT)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd", self.scalar_type());
  }

#undef ANY_ALL_OUT
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
