/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>
#include <cstring>

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

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

  for (size_t i = 0; i < out.dim(); ++i) {
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

  ET_KERNEL_CHECK(
      ctx, check_unsqueeze_copy_args(self, dim, out), InvalidArgument, out);

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
