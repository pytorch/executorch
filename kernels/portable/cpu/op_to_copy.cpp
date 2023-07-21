// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

template <typename SELF_CTYPE, typename OUT_CTYPE>
void _to_impl(const Tensor& self, Tensor& out) {
  auto self_data = self.data_ptr<SELF_CTYPE>();
  auto out_data = out.data_ptr<OUT_CTYPE>();

  for (int i = 0; i < self.numel(); i++) {
    out_data[i] = static_cast<OUT_CTYPE>(self_data[i]);
  }
}

// to_copy.out(Tensor self, *, bool non_blocking=False, MemoryFormat?
// memory_format=None, Tensor(a!) out) -> Tensor(a!)
Tensor& to_copy_out(
    RuntimeContext& context,
    const Tensor& self,
    bool non_blocking,
    exec_aten::optional<exec_aten::MemoryFormat> memory_format,
    Tensor& out) {
  // Right now we only support blocking data transfer
  ET_CHECK(non_blocking == false);

  // Right now we only focus on contiguous memory, memory_format shall be
  // exec::aten::MemoryFormat::Contiguous or none.
  ET_CHECK(
      !memory_format.has_value() ||
      memory_format.value() == MemoryFormat::Contiguous);

  torch::executor::Error err = resize_tensor(out, self.sizes());
  ET_CHECK_MSG(
      err == torch::executor::Error::Ok,
      "Failed to resize out Tensor in to_copy_out");

  // self and out should be in same size.
  ET_CHECK_SAME_SHAPE2(self, out);
// Use a two layer switch to hanldle each possible data pairs
#define TO_IMPL(SELF_CTYPE, OUT_CTYPE, out_dtype) \
  case ScalarType::out_dtype:                     \
    _to_impl<SELF_CTYPE, OUT_CTYPE>(self, out);   \
    break;

#define CASE_TENSOR_DTYPE(SELF_CTYPE, self_dtype)                              \
  case ScalarType::self_dtype:                                                 \
    switch (out.scalar_type()) {                                               \
      ET_FORALL_REAL_TYPES_AND_WITH(Bool, SELF_CTYPE, TO_IMPL)                 \
      default:                                                                 \
        ET_CHECK_MSG(false, "Unhandled output dtype %hhd", out.scalar_type()); \
    }                                                                          \
    break;

  switch (self.scalar_type()) {
    ET_FORALL_REAL_TYPES_AND(Bool, CASE_TENSOR_DTYPE);
    default:
      ET_CHECK_MSG(false, "Unhandled input dtype %hhd", self.scalar_type());
  }

#undef CASE_TENSOR_DTYPE
#undef TO_IMPL

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
