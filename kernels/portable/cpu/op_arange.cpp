/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/arange_util.h>
#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace torch {
namespace executor {
namespace native {

Tensor& arange_out(KernelRuntimeContext& ctx, const Scalar& end, Tensor& out) {
  double end_val = 0;
  ET_KERNEL_CHECK(
      ctx, utils::extract_scalar(end, &end_val), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, check_arange_args(0.0, end_val, 1.0, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(out), InvalidArgument, out);

  Tensor::SizesType out_length = compute_arange_out_size(0.0, end_val, 1.0);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {&out_length, 1}) == Error::Ok,
      InvalidArgument,
      out);

  arange_out_impl(ctx, end_val, out);

  return out;
}

Tensor& arange_start_out(
    KernelRuntimeContext& ctx,
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& out) {
  (void)ctx;

  double d_start = 0;
  ET_KERNEL_CHECK(
      ctx, utils::extract_scalar(start, &d_start), InvalidArgument, out);

  double d_end = 0;
  ET_KERNEL_CHECK(
      ctx, utils::extract_scalar(end, &d_end), InvalidArgument, out);

  double d_step = 0;
  ET_KERNEL_CHECK(
      ctx, utils::extract_scalar(step, &d_step), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx,
      check_arange_args(d_start, d_end, d_step, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(out), InvalidArgument, out);

  Tensor::SizesType out_length =
      compute_arange_out_size(d_start, d_end, d_step);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {&out_length, 1}) == Error::Ok,
      InvalidArgument,
      out);

  arange_out_impl(ctx, d_start, d_end, d_step, out);

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
