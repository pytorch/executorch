/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/slice_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <cstring>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

Tensor& slice_copy_Tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    int64_t dim,
    exec_aten::optional<int64_t> start_val,
    exec_aten::optional<int64_t> end_val,
    int64_t step,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, check_slice_copy_args(in, dim, step, out), InvalidArgument, out);

  if (dim < 0) {
    dim += in.dim();
  }

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  // If user do not set value to end_val, set end to in.size(dim) (largest
  // value available)
  int64_t end = end_val.has_value() ? end_val.value() : in.size(dim);
  // If user do not set value to start_val, set start to 0 (smallest value
  // available)
  int64_t start = start_val.has_value() ? start_val.value() : 0;

  int64_t length = adjust_slice_indices(in.size(dim), &start, &end, step);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  Tensor::SizesType target_sizes[kTensorDimensionLimit];
  size_t target_ndim = 0;
  get_slice_copy_out_target_size(in, dim, length, target_sizes, &target_ndim);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {target_sizes, target_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  compute_slice(in, dim, start, length, step, out);

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
