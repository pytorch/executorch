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

using Tensor = exec_aten::Tensor;

bool check_narrow_copy_args(
    const Tensor& in,
    int64_t dim,
    int64_t start,
    int64_t lenth,
    Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(in.dim() > 0);
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(in, dim));
  ET_LOG_MSG_AND_RETURN_IF_FALSE(lenth >= 0, "lenth must be non-negative");
  ET_LOG_AND_RETURN_IF_FALSE(start >= -in.size(dim));
  ET_LOG_AND_RETURN_IF_FALSE(start <= in.size(dim));
  if (start < 0) {
    start += in.size(dim);
  }
  ET_LOG_AND_RETURN_IF_FALSE(start + lenth <= in.size(dim));
  return true;
}

void get_narrow_copy_out_target_size(
    const Tensor& in,
    int64_t dim,
    int64_t length,
    exec_aten::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = in.dim();

  for (size_t d = 0; d < in.dim(); ++d) {
    out_sizes[d] = in.size(d);
  }
  out_sizes[dim] = length;
}

bool check_slice_copy_args(
    const Tensor& in,
    int64_t dim,
    int64_t step,
    Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(in.dim() > 0);
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(in, dim));
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      step > 0, "slice step must be greater than zero");
  return true;
}

void get_slice_copy_out_target_size(
    const Tensor& in,
    int64_t dim,
    int64_t length,
    exec_aten::SizesType* out_sizes,
    size_t* out_ndim) {
  get_narrow_copy_out_target_size(in, dim, length, out_sizes, out_ndim);
}

bool check_slice_scatter_args(
    const Tensor& input,
    const Tensor& src,
    int64_t dim,
    int64_t num_values,
    int64_t step,
    Tensor output) {
  ET_LOG_AND_RETURN_IF_FALSE(input.dim() > 0);

  // Check dim. The dim planed to be selected on shall exist in input
  ET_LOG_AND_RETURN_IF_FALSE(dim_is_valid(dim, input.dim()));

  // Input and output tensors should be the same shape and dtype
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_shape_and_dtype(input, output));

  // The input.dim() shall equal to src.dim()
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_rank(input, src));

  // Check step. Step must be greater than zero
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      step > 0, "slice step must be greater than zero");

  // The size of src tensor should follow these rules:
  // - src.size(i) shall equal to input.size(i) if i != dim,
  // - src.size(dim) shall equal to num_values
  for (size_t d = 0; d < input.dim() - 1; d++) {
    if (d != dim) {
      ET_LOG_AND_RETURN_IF_FALSE(
          tensors_have_same_size_at_dims(input, d, src, d));
    } else {
      ET_LOG_MSG_AND_RETURN_IF_FALSE(
          src.size(d) == num_values,
          "input.size(%zu) %zd != num_values %" PRId64 " | dim = %" PRId64 ")",
          d,
          input.size(d),
          num_values,
          dim);
    }
  }

  return true;
}

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

void compute_slice(
    const Tensor& in,
    int64_t dim,
    int64_t start,
    int64_t length,
    int64_t step,
    Tensor& out) {
  size_t dim_length = in.size(dim);

  size_t leading_dims = getLeadingDims(in, dim);
  size_t trailing_dims = getTrailingDims(in, dim);

  if (trailing_dims == 0) {
    return;
  }

  size_t length_per_step = trailing_dims * in.element_size();

  const char* input_data = in.const_data_ptr<char>();
  char* dest = out.mutable_data_ptr<char>();

  for (int i = 0; i < leading_dims; i++) {
    const char* src = input_data + (i * dim_length + start) * length_per_step;
    for (int j = 0; j < length; j++) {
      memcpy(dest, src, length_per_step);
      src += step * length_per_step;
      dest += length_per_step;
    }
  }
}

} // namespace executor
} // namespace torch
