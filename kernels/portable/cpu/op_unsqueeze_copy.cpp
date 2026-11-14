/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <cstdint>
#include <cstring>

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;

// unsqueeze_copy.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)
// -> Tensor(a!)
Tensor& unsqueeze_copy_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    int64_t dim,
    Tensor& out) {
  (void)ctx;
  Tensor::SizesType expected_output_size[kTensorDimensionLimit];
  // I think this is safe to do but need to confirm.
  // If we can do this then subsequent checks that specialize on dim < 0
  // are not needed
  if (dim < 0) {
    dim += out.dim();
    ET_KERNEL_CHECK(ctx, dim >= 0, InvalidArgument, out);
  }

  ET_KERNEL_CHECK(ctx, self.dim() + 1 == out.dim(), InvalidArgument, out);
  ET_KERNEL_CHECK(ctx, dim <= self.dim(), InvalidArgument, out);

  for (const auto i : c10::irange(out.dim())) {
    if (i < dim) {
      expected_output_size[i] = self.size(i);
    } else if (i > dim) {
      expected_output_size[i] = self.size(i - 1);
    } else {
      expected_output_size[i] = 1;
    }
  }

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(
          out, {expected_output_size, static_cast<size_t>(out.dim())}) ==
          Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, check_unsqueeze_copy_args(self, dim, out), InvalidArgument, out);

  if (self.nbytes() == 0) {
    // Note that this check is important. It's valid for a tensor with numel 0
    // to have a null data pointer, but in some environments it's invalid to
    // pass a null pointer to memcpy() even when the size is zero.
    return out;
  }
  if (is_contiguous_dim_order(
          self.dim_order().data(), self.dim_order().size())) {
    memcpy(out.mutable_data_ptr(), self.const_data_ptr(), self.nbytes());
  } else {
    // self can be in another dim order, like the result of nonzero, while out
    // is contiguous, as PyTorch's unsqueeze_copy returns it. Copy the elements
    // in logical order.
    const auto* input_data = self.const_data_ptr<uint8_t>();
    auto* output_data = out.mutable_data_ptr<uint8_t>();
    const auto element_size = self.element_size();
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
