// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstring>

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

Tensor& alias_copy_out(RuntimeContext& context, const Tensor& in, Tensor& out) {
  (void)context;

  ET_CHECK(resize_tensor(out, in.sizes()) == torch::executor::Error::Ok);
  ET_CHECK_SAME_DTYPE2(in, out);

  if (in.nbytes() > 0) {
    // Note that this check is important. It's valid for a tensor with numel 0
    // to have a null data pointer, but in some environments it's invalid to
    // pass a null pointer to memcpy() even when the size is zero.
    memcpy(out.mutable_data_ptr(), in.const_data_ptr(), in.nbytes());
  }
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
