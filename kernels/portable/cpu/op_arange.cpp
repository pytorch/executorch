/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace torch {
namespace executor {
namespace native {

Tensor& arange_out(RuntimeContext& ctx, const Scalar& end, Tensor& out) {
  ET_KERNEL_CHECK_MSG(
      ctx,
      out.dim() == 1,
      InvalidArgument,
      out,
      "out should be a 1-d tensor, but got a %zu-d tensor",
      out.dim());

  ScalarType end_type = utils::get_scalar_dtype(end);

  double end_val = 0;
  ET_SWITCH_SCALAR_OBJ_TYPES(end_type, ctx, __func__, CTYPE_END, [&]() {
    CTYPE_END end_v;
    ET_EXTRACT_SCALAR(end, end_v);
    ET_KERNEL_CHECK_MSG(
        ctx,
        end_v >= 0,
        InvalidArgument,
        out,
        "Input end should be non-negative.");
    end_val = static_cast<double>(end_v);
  });

  size_t size = static_cast<size_t>(std::ceil(end_val));

  Tensor::SizesType out_length = static_cast<Tensor::SizesType>(size);
  Error status = resize_tensor(out, {&out_length, 1});
  ET_KERNEL_CHECK_MSG(
      ctx, status == Error::Ok, InvalidArgument, out, "resize_tensor fails");

  ET_SWITCH_REAL_TYPES(out.scalar_type(), ctx, __func__, CTYPE, [&]() {
    auto out_data = out.mutable_data_ptr<CTYPE>();
    for (size_t i = 0; i < size; i++) {
      out_data[i] = static_cast<CTYPE>(i);
    }
  });

  return out;
}

Tensor& arange_start_out(
    RuntimeContext& ctx,
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& out) {
  (void)ctx;

  ScalarType start_type = utils::get_scalar_dtype(start);
  ScalarType end_type = utils::get_scalar_dtype(end);
  ScalarType step_type = utils::get_scalar_dtype(step);

  double d_start = 0;
  ET_SWITCH_SCALAR_OBJ_TYPES(start_type, ctx, __func__, CTYPE_END, [&]() {
    CTYPE_END start_v;
    ET_EXTRACT_SCALAR(start, start_v);
    d_start = static_cast<double>(start_v);
  });

  double d_end = 0;
  ET_SWITCH_SCALAR_OBJ_TYPES(end_type, ctx, __func__, CTYPE_END, [&]() {
    CTYPE_END end_v;
    ET_EXTRACT_SCALAR(end, end_v);
    d_end = static_cast<double>(end_v);
  });

  double d_step = 0;
  ET_SWITCH_SCALAR_OBJ_TYPES(step_type, ctx, __func__, CTYPE_END, [&]() {
    CTYPE_END step_v;
    ET_EXTRACT_SCALAR(step, step_v);
    d_step = static_cast<double>(step_v);
  });

  ET_KERNEL_CHECK_MSG(
      ctx,
      (d_step > 0 && (d_end >= d_start)) || (d_step < 0 && (d_end <= d_start)),
      InvalidArgument,
      out,
      "upper bound and larger bound inconsistent with step sign");

  double size_d = (d_end - d_start) / d_step;
  size_t size = static_cast<size_t>(std::ceil(size_d));

  Tensor::SizesType out_length = static_cast<Tensor::SizesType>(size);
  Error status = resize_tensor(out, {&out_length, 1});
  ET_KERNEL_CHECK_MSG(
      ctx, status == Error::Ok, InvalidArgument, out, "resize_tensor fails");

  ET_SWITCH_REAL_TYPES(out.scalar_type(), ctx, __func__, CTYPE, [&]() {
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
