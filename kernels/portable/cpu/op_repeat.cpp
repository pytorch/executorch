/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <executorch/kernels/portable/cpu/util/repeat_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {
namespace {

bool calculate_output_size(
    const exec_aten::ArrayRef<exec_aten::SizesType>& self_sizes,
    const exec_aten::ArrayRef<int64_t>& repeats,
    Tensor::SizesType* out_sizes_ptr) {
  ET_LOG_AND_RETURN_IF_FALSE(repeats.size() < kTensorDimensionLimit);

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      repeats.size() >= self_sizes.size(),
      "Repeats vector size is %zu must be >= self_sizes %zu.",
      repeats.size(),
      self_sizes.size());

  int32_t i = 0;
  for (; i < (repeats.size() - self_sizes.size()); ++i) {
    out_sizes_ptr[i] = static_cast<exec_aten::SizesType>(repeats[i]);
  }
  int32_t j = 0;
  for (; i < repeats.size(); ++i) {
    out_sizes_ptr[i] =
        static_cast<exec_aten::SizesType>(repeats[i]) * self_sizes[j];
    j++;
  }

  return true;
}

} // namespace

using Tensor = exec_aten::Tensor;

// repeat.out(Tensor self, int[] repeats, *, Tensor(a!) out) -> Tensor(a!)
Tensor& repeat_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    exec_aten::ArrayRef<int64_t> repeats,
    Tensor& out) {
  (void)ctx;
  Tensor::SizesType expected_output_size[kTensorDimensionLimit];

  ET_KERNEL_CHECK(
      ctx,
      calculate_output_size(self.sizes(), repeats, expected_output_size),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(self, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(self), InvalidArgument, out);

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, {expected_output_size, repeats.size()}) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ET_KERNEL_CHECK(
      ctx,
      repeat_tensor(self, repeats, out) == Error::Ok,
      InvalidArgument,
      out);

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
