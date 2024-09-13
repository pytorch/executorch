/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/transpose_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using SizesType = exec_aten::SizesType;
using StridesType = exec_aten::StridesType;
using Tensor = exec_aten::Tensor;

/**
 * Swaps dimension 'dim0' of 'a' with 'dim1', and copying
 * that mutation into `out` in a manner such that the data is densely packed
 * and is_contiguous() would return true (stride dim[size-1] = 1).
 *
 * transpose_copy.int_out(Tensor self, int dim0, int dim1, *, Tensor(a!) out)
 */
Tensor& transpose_copy_int_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    int64_t dim0,
    int64_t dim1,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_transpose_copy_args(in, dim0, dim1, out),
      InvalidArgument,
      out);

  if (dim0 < 0) {
    dim0 += nonzero_dim(in);
  }
  if (dim1 < 0) {
    dim1 += nonzero_dim(in);
  }

  Tensor::SizesType expected_out_size[kTensorDimensionLimit];
  size_t expected_out_dim = 0;
  get_transpose_out_target_size(
      in, dim0, dim1, expected_out_size, &expected_out_dim);

  // Resize for dynamic shape
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {expected_out_size, expected_out_dim}) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_SWITCH_ALL_TYPES(in.scalar_type(), ctx, __func__, CTYPE, [&] {
    transpose_tensors<CTYPE>(in, dim0, dim1, out);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
