/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
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

  size_t size = static_cast<size_t>(std::ceil(end_val));

  Tensor::SizesType out_length = static_cast<Tensor::SizesType>(size);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {&out_length, 1}) == Error::Ok,
      InvalidArgument,
      out);

  ET_SWITCH_REAL_TYPES(out.scalar_type(), ctx, "arange.out", CTYPE, [&]() {
    auto out_data = out.mutable_data_ptr<CTYPE>();
    for (size_t i = 0; i < size; i++) {
      out_data[i] = static_cast<CTYPE>(i);
    }
  });

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

  double size_d = (d_end - d_start) / d_step;
  size_t size = static_cast<size_t>(std::ceil(size_d));

  Tensor::SizesType out_length = static_cast<Tensor::SizesType>(size);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {&out_length, 1}) == Error::Ok,
      InvalidArgument,
      out);

  ET_SWITCH_REAL_TYPES(
      out.scalar_type(), ctx, "arange.start_out", CTYPE, [&]() {
        auto out_data = out.mutable_data_ptr<CTYPE>();
        for (size_t i = 0; i < size; i++) {
          out_data[i] = convert<CTYPE, double>(d_start + i * d_step);
        }
      });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
