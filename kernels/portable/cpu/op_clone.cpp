/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

// clone.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out)
// -> Tensor(a!)
Tensor& clone_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    exec_aten::optional<exec_aten::MemoryFormat> memory_format,
    Tensor& out) {
  (void)context;

  ET_KERNEL_CHECK(
      context,
      resize_tensor(out, self.sizes()) == torch::executor::Error::Ok,
      InvalidArgument,
      out);

  // The input and out shall share same dtype and size
  ET_KERNEL_CHECK(
      context,
      tensors_have_same_shape_and_dtype(self, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      context, tensors_have_same_dim_order(self, out), InvalidArgument, out);

  // Right now we only focus on contiguous memory, memory_format shall always
  // either a nullopt or exec::aten::MemoryFormat::Contiguous
  ET_KERNEL_CHECK(
      context,
      !memory_format.has_value() ||
          memory_format.value() == MemoryFormat::Contiguous,
      InvalidArgument,
      out);

  if (self.nbytes() > 0) {
    // Note that this check is important. It's valid for a tensor with numel 0
    // to have a null data pointer, but in some environments it's invalid to
    // pass a null pointer to memcpy() even when the size is zero.
    memcpy(out.mutable_data_ptr(), self.const_data_ptr(), self.nbytes());
  }
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
