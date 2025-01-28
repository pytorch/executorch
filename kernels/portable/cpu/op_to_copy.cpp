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
    KernelRuntimeContext& ctx,
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

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(self, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(self), InvalidArgument, out);

  ET_SWITCH_REALHBBF16_TYPES(self.scalar_type(), ctx, "to_copy", CTYPE_IN, [&] {
    ET_SWITCH_REALHBBF16_TYPES(
        out.scalar_type(), ctx, "to_copy", CTYPE_OUT, [&] {
          _to_impl<CTYPE_IN, CTYPE_OUT>(self, out);
        });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
