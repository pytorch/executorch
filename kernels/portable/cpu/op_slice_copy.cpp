/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <cstring>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

namespace {

int64_t adjust_slice_indices(
    int64_t dim_length,
    int64_t* start,
    int64_t* end,
    int64_t step) {
  int64_t num_values = 0;

  // Update start and end index
  // First convert it to c++ style from python style if needed.
  // The start index is using python style E.g., for the shape {2, 3, 4},
  // dim = -1 would refer to dim[2], dim = -2 would refer to dim[1], and so on.
  *start = *start < 0 ? *start + dim_length : *start;
  *end = *end < 0 ? *end + dim_length : *end;
  // Second, if start or end still negative, which means user want to start or
  // end slicing from very beginning, so set it to zero
  *start = *start < 0 ? 0 : *start;
  *end = *end < 0 ? 0 : *end;
  // Last, if start or end larger than maximum value (dim_length - 1), indicates
  // user want to start slicing after end or slicing until the end, so update it
  // to dim_length
  *start = *start > dim_length ? dim_length : *start;
  *end = *end > dim_length ? dim_length : *end;

  if (*start >= dim_length || *end <= 0 || *start >= *end) {
    // Set num_values to 0 if interval [start, end) is non-exist or do not
    // overlap with [0, dim_length)
    num_values = 0;
  } else {
    // Update num_values to min(max_num_values, num_values)
    num_values = (*end - 1 - *start) / step + 1;
  }
  return num_values;
}

} // namespace

Tensor& slice_copy_Tensor_out(
    RuntimeContext& ctx,
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

  // If user do not set value to end_val, set end to in.size(dim) (largest
  // value available)
  int64_t end = end_val.has_value() ? end_val.value() : in.size(dim);
  // If user do not set value to start_val, set start to 0 (smallest value
  // available)
  int64_t start = start_val.has_value() ? start_val.value() : 0;

  int64_t num_values = adjust_slice_indices(in.size(dim), &start, &end, step);

  Tensor::SizesType target_sizes[kTensorDimensionLimit];
  size_t target_ndim = 0;
  get_slice_copy_out_target_size(
      in, dim, num_values, target_sizes, &target_ndim);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {target_sizes, target_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  size_t dim_length = in.size(dim);

  size_t leading_dims = getLeadingDims(in, dim);
  size_t trailing_dims = getTrailingDims(in, dim);

  if (trailing_dims == 0) {
    return out;
  }

  size_t length_per_step = trailing_dims * in.element_size();

  const char* input_data = in.const_data_ptr<char>();
  char* dest = out.mutable_data_ptr<char>();

  for (int i = 0; i < leading_dims; i++) {
    const char* src = input_data + (i * dim_length + start) * length_per_step;
    for (int j = 0; j < num_values; j++) {
      memcpy(dest, src, length_per_step);
      src += step * length_per_step;
      dest += length_per_step;
    }
  }
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
