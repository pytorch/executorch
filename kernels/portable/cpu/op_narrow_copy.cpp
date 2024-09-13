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

Tensor& narrow_copy_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    int64_t dim,
    int64_t start,
    int64_t length,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_narrow_copy_args(in, dim, start, length, out),
      InvalidArgument,
      out);

  if (dim < 0) {
    dim += in.dim();
  }

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  Tensor::SizesType target_sizes[kTensorDimensionLimit];
  size_t target_ndim = 0;
  get_narrow_copy_out_target_size(in, dim, length, target_sizes, &target_ndim);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {target_sizes, target_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  if (length != 0) {
    compute_slice(in, dim, start, length, 1, out);
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
