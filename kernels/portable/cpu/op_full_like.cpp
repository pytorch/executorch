// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <executorch/kernels/kernel_includes.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>

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
void full_like_kernel(const Scalar& value, Tensor& out) {
  CTYPE value_v;
  bool ok = utils::extract_scalar(value, &value_v);
  ET_CHECK_MSG(ok, "Invalid fill value: wrong type or out of range");
  const size_t n = out.numel();
  auto data_out = out.data_ptr<CTYPE>();
  for (size_t i = 0; i < n; ++i) {
    data_out[i] = value_v;
  }
}
} // namespace

/**
 * Returns a tensor with the same size as input filled with fill_value
 */
Tensor& full_like_out(
    RuntimeContext& context,
    const Tensor& self,
    const Scalar& fill_value,
    optional<MemoryFormat> memory_format,
    Tensor& out) {
  (void)context;

  torch::executor::Error err = resize_tensor(out, self.sizes());
  ET_CHECK_MSG(
      err == torch::executor::Error::Ok,
      "Failed to resize out Tensor in full_like_out");

  ET_CHECK_SAME_SHAPE2(self, out);
  if (memory_format.has_value()) {
    ET_CHECK_MSG(
        memory_format.value() == MemoryFormat::Contiguous,
        "memory_format must be contiguous");
  }

#define FULL_LIKE(ctype, dtype)               \
  case ScalarType::dtype:                     \
    full_like_kernel<ctype>(fill_value, out); \
    break;

  switch (out.scalar_type()) {
    ET_FORALL_REAL_TYPES_AND(Bool, FULL_LIKE)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd", self.scalar_type());
  }

#undef FULL_LIKE

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
