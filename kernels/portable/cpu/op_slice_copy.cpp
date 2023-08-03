/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>
#include <cstring>

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

namespace {

// TODO(gasoonjia): Move this to a common spot so all implementation of
// this operator can share it. (e.g., DSP-specific)
/// Asserts that the parameters are valid.
void check_slice_copy_Tensor_out_args(
    const Tensor input,
    int64_t dim,
    int64_t num_values,
    int64_t step,
    Tensor output) {
  //
  // Check dim. The dim planed to be selected on shall exist in input
  ET_CHECK_MSG(
      dim >= 0 && dim < input.dim(),
      "dim %" PRId64 " out of range [0,%zd)",
      dim,
      input.dim());

  // Input dtype shall match the output dtype.
  ET_CHECK_SAME_DTYPE2(input, output);

  // The output.dim() shall equal to input.dim(), based on the definition of
  // slicing.
  ET_CHECK_MSG(
      input.dim() == output.dim(),
      "input.dim() %zd != output.dim() %zd",
      input.dim(),
      output.dim());

  // Check step. Step must be greater than zero
  ET_CHECK_MSG(step > 0, "slice step must be greater than zero");

  // The size of output tensor should follow these rules:
  // - output.size(i) shall equal to input.size(i) if i != dim,
  // - output.size(dim) shall equal to num_values
  for (size_t d = 0; d < input.dim() - 1; d++) {
    if (d != dim) {
      ET_CHECK_MSG(
          input.size(d) == output.size(d),
          "input.size(%zu) %zd != output.size(%zu) %zd | dim = %" PRId64 ")",
          d,
          input.size(d),
          d,
          output.size(d),
          dim);
    } else {
      ET_CHECK_MSG(
          output.size(d) == num_values,
          "input.size(%zu) %zd != num_values %" PRId64 " | dim = %" PRId64 ")",
          d,
          input.size(d),
          num_values,
          dim);
    }
  }
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

} // namespace

/// slice_copy.Tensor_out(Tensor self, int dim=0, int? start=None, int?
/// end=None, int step=1, *, Tensor(a!) out) -> Tensor(a!)
/// -> Tensor(a!)
Tensor& slice_copy_Tensor_out(
    RuntimeContext& context,
    const Tensor& input,
    int64_t dim,
    exec_aten::optional<int64_t> start_val,
    exec_aten::optional<int64_t> end_val,
    int64_t step,
    Tensor& out) {
  (void)context;
  if (dim < 0) {
    dim += input.dim();
  }

  // If user do not set value to end_val, set end to input.size(dim) (largest
  // value available)
  int64_t end = end_val.has_value() ? end_val.value() : input.size(dim);
  // If user do not set value to start_val, set start to 0 (smallest value
  // available)
  int64_t start = start_val.has_value() ? start_val.value() : 0;

  int64_t num_values =
      adjust_slice_indices(input.size(dim), &start, &end, step);

  check_slice_copy_Tensor_out_args(input, dim, num_values, step, out);

  size_t dim_length = input.size(dim);

  size_t leading_dims = getLeadingDims(input, dim);
  size_t trailing_dims = getTrailingDims(input, dim);

  size_t length_per_step = trailing_dims * input.element_size();

  const char* input_data = input.const_data_ptr<char>();
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
