/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;

// clone.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out)
// -> Tensor(a!)
Tensor& clone_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    std::optional<exec_aten::MemoryFormat> memory_format,
    Tensor& out) {
  (void)context;

  ET_KERNEL_CHECK(
      context,
      resize_tensor(out, self.sizes()) == torch::executor::Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      context,
      tensors_have_same_shape_and_dtype(self, out),
      InvalidArgument,
      out);

  if (!tensors_have_same_dim_order(self, out)) {
    ET_LOG(
        Error,
        "op_clone.out: dim_order mismatch: self.dtype=%d out.dtype=%d. "
        "See github.com/pytorch/executorch/issues/16032",
        (int)self.scalar_type(),
        (int)out.scalar_type());
  }
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
    memcpy(out.mutable_data_ptr(), self.const_data_ptr(), self.nbytes());
  }
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
