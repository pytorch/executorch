/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {
namespace {

bool check_repeat_interleave_args(
    const Tensor& repeats,
    int64_t output_size_value,
    int64_t repeats_sum,
    Tensor& out) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      repeats.scalar_type() == ScalarType::Int ||
          repeats.scalar_type() == ScalarType::Long,
      "repeats must be int or long");
  ET_LOG_MSG_AND_RETURN_IF_FALSE(repeats.dim() == 1, "repeats must be 1D");
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      output_size_value == repeats_sum,
      "output_size, if provided, must be equal to repeats.sum()");
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(repeats, out));

  if (repeats.scalar_type() == ScalarType::Long) {
    const int64_t* const repeats_data = repeats.const_data_ptr<int64_t>();
    for (size_t i = 0; i < repeats.numel(); ++i) {
      ET_LOG_MSG_AND_RETURN_IF_FALSE(
          repeats_data[i] >= 0, "repeats cannot be negative");
    }
  } else {
    const int32_t* const repeats_data = repeats.const_data_ptr<int32_t>();
    for (size_t i = 0; i < repeats.numel(); ++i) {
      ET_LOG_MSG_AND_RETURN_IF_FALSE(
          repeats_data[i] >= 0, "repeats cannot be negative");
    }
  }

  return true;
}

} // namespace

using Tensor = exec_aten::Tensor;

Tensor& repeat_interleave_Tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& repeats,
    exec_aten::optional<int64_t> output_size,
    Tensor& out) {
  (void)ctx;

  int64_t repeats_sum = 0;

  constexpr auto name = "repeat_interleave.Tensor_out";

  ET_SWITCH_TWO_TYPES(Int, Long, repeats.scalar_type(), ctx, name, CTYPE, [&] {
    const CTYPE* repeats_data = repeats.const_data_ptr<CTYPE>();
    for (size_t ix = 0; ix < repeats.numel(); ++ix) {
      repeats_sum += static_cast<int64_t>(repeats_data[ix]);
    }
  });

  int64_t output_size_value =
      output_size.has_value() ? output_size.value() : repeats_sum;

  ET_KERNEL_CHECK(
      ctx,
      check_repeat_interleave_args(
          repeats, output_size_value, repeats_sum, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(repeats, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensor_is_default_dim_order(repeats), InvalidArgument, out);

  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(
          out, {static_cast<exec_aten::SizesType>(output_size_value)}) ==
          Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ET_SWITCH_TWO_TYPES(Int, Long, repeats.scalar_type(), ctx, name, CTYPE, [&] {
    const CTYPE* repeats_data = repeats.const_data_ptr<CTYPE>();
    CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
    size_t out_ix = 0;
    for (size_t ix = 0; ix < repeats.numel(); ix++) {
      for (CTYPE i = 0; i < repeats_data[ix]; i++, out_ix++) {
        out_data[out_ix] = static_cast<CTYPE>(ix);
      }
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
