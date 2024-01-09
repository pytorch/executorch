/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

template <typename SELF_CTYPE, typename OUT_CTYPE>
void _to_impl(const Tensor& self, Tensor& out) {
  auto self_data = self.mutable_data_ptr<SELF_CTYPE>();
  auto out_data = out.mutable_data_ptr<OUT_CTYPE>();

  for (int i = 0; i < self.numel(); i++) {
    out_data[i] = static_cast<OUT_CTYPE>(self_data[i]);
  }
}

// to_copy.out(Tensor self, *, bool non_blocking=False, MemoryFormat?
// memory_format=None, Tensor(a!) out) -> Tensor(a!)
Tensor& to_copy_out(
    RuntimeContext& ctx,
    const Tensor& self,
    bool non_blocking,
    exec_aten::optional<exec_aten::MemoryFormat> memory_format,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      check_to_copy_args(self, non_blocking, memory_format, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, self.sizes()) == torch::executor::Error::Ok,
      InvalidArgument,
      out);

// Use a two layer switch to hanldle each possible data pairs
#define TO_IMPL(SELF_CTYPE, OUT_CTYPE, out_dtype) \
  case ScalarType::out_dtype:                     \
    _to_impl<SELF_CTYPE, OUT_CTYPE>(self, out);   \
    break;

#define CASE_TENSOR_DTYPE(SELF_CTYPE, self_dtype)              \
  case ScalarType::self_dtype:                                 \
    switch (out.scalar_type()) {                               \
      ET_FORALL_REAL_TYPES_AND_WITH(Bool, SELF_CTYPE, TO_IMPL) \
      default:                                                 \
        ET_CHECK_MSG(                                          \
            false,                                             \
            "Unhandled output dtype %" PRId8,                  \
            static_cast<int8_t>(out.scalar_type()));           \
    }                                                          \
    break;

  switch (self.scalar_type()) {
    ET_FORALL_REAL_TYPES_AND(Bool, CASE_TENSOR_DTYPE);
    default:
      ET_CHECK_MSG(
          false,
          "Unhandled input dtype %" PRId8,
          static_cast<int8_t>(self.scalar_type()));
  }

#undef CASE_TENSOR_DTYPE
#undef TO_IMPL

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
