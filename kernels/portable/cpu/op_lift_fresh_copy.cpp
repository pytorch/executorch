// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstring>

#include <executorch/kernels/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

Tensor&
lift_fresh_copy_out(RuntimeContext& context, const Tensor& self, Tensor& out) {
  (void)context;
  // The input and out shall share same dtype and size
  ET_CHECK_SAME_SHAPE_AND_DTYPE2(self, out);

  if (self.nbytes() > 0) {
    // Note that this check is important. It's valid for a tensor with numel 0
    // to have a null data pointer, but in some environments it's invalid to
    // pass a null pointer to memcpy() even when the size is zero.
    memcpy(out.data_ptr(), self.data_ptr(), self.nbytes());
  }
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
