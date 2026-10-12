/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>
#include <cstring>

#include <executorch/kernels/portable/cpu/util/broadcast_indexes_range.h>
#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;

// view_copy.out(Tensor self, int[] size, *, Tensor(a!) out) -> Tensor(a!)
Tensor& view_copy_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    executorch::aten::ArrayRef<int64_t> size_int64_t,
    Tensor& out) {
  (void)ctx;

  Tensor::SizesType expected_output_size[16];
  ET_KERNEL_CHECK(
      ctx,
      get_view_copy_target_size(
          self, size_int64_t, out.dim(), expected_output_size),
      InvalidArgument,
      out);

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(
          out, {expected_output_size, static_cast<size_t>(out.dim())}) ==
          Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, check_view_copy_args(self, size_int64_t, out), InvalidArgument, out);

  if (self.nbytes() == 0) {
    return out;
  }
  if (tensor_is_default_dim_order(self)) {
    memcpy(out.mutable_data_ptr(), self.const_data_ptr(), self.nbytes());
  } else {
    const auto* input_data = self.const_data_ptr<uint8_t>();
    auto* output_data = out.mutable_data_ptr<uint8_t>();
    const auto element_size = self.element_size();
    // Iterate the input's logical shape; the output may have a different rank.
    for (const auto [output_index, input_index] :
         BroadcastIndexesRange<1, /*support_noncontiguous_input_tensors=*/true>(
             self, self)) {
      memcpy(
          output_data + output_index * element_size,
          input_data + input_index * element_size,
          element_size);
    }
  }
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
