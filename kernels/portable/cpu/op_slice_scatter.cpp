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
void check_input_args(
    const Tensor& input,
    const Tensor& src,
    int64_t dim,
    int64_t num_values,
    int64_t step,
    Tensor output) {
  // Check dim. The dim planed to be selected on shall exist in input
  ET_CHECK_MSG(
      dim >= 0 && dim < input.dim(),
      "dim %" PRId64 " out of range [0,%zd)",
      dim,
      input.dim());

  // Input and output tensors should be the same shape
  ET_CHECK_SAME_SHAPE2(input, output);

  // Input and output tensors should have the same shape
  ET_CHECK_SAME_DTYPE2(input, output);

  // The input.dim() shall equal to src.dim()
  ET_CHECK_MSG(
      input.dim() == src.dim(),
      "input.dim() %zd != src.dim() %zd",
      input.dim(),
      src.dim());

  // Check step. Step must be greater than zero
  ET_CHECK_MSG(step > 0, "slice step must be greater than zero");

  // The size of src tensor should follow these rules:
  // - src.size(i) shall equal to input.size(i) if i != dim,
  // - src.size(dim) shall equal to num_values
  for (size_t d = 0; d < input.dim() - 1; d++) {
    if (d != dim) {
      ET_CHECK_MSG(
          input.size(d) == src.size(d),
          "input.size(%zu) %zd != src.size(%zu) %zd | dim = %" PRId64 ")",
          d,
          input.size(d),
          d,
          src.size(d),
          dim);
    } else {
      ET_CHECK_MSG(
          src.size(d) == num_values,
          "input.size(%zu) %zd != num_values %" PRId64 " | dim = %" PRId64 ")",
          d,
          input.size(d),
          num_values,
          dim);
    }
  }
}

// Output tensor should be the same size as the input tensor
Error resize_like_input(const Tensor& input, Tensor& out) {
  Tensor::SizesType expected_output_size[kTensorDimensionLimit];
  for (size_t i = 0; i < input.dim(); ++i) {
    expected_output_size[i] = input.size(i);
  }
  ArrayRef<Tensor::SizesType> output_size{
      expected_output_size, static_cast<size_t>(input.dim())};
  auto error = resize_tensor(out, output_size);

  return error;
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

Tensor& slice_scatter_out(
    RuntimeContext& ctx,
    const Tensor& input,
    const Tensor& src,
    int64_t dim,
    exec_aten::optional<int64_t> start_val,
    exec_aten::optional<int64_t> end_val,
    int64_t step,
    Tensor& out) {
  (void)ctx;

  if (dim < 0) {
    dim += input.dim();
  }

  // resize out tensor for dynamic shapes
  auto error = resize_like_input(input, out);
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");

  // If user do not set value to end_val, set end to input.size(dim) (largest
  // value available)
  int64_t end = end_val.has_value() ? end_val.value() : input.size(dim);
  // If user do not set value to start_val, set start to 0 (smallest value
  // available)
  int64_t start = start_val.has_value() ? start_val.value() : 0;

  int64_t num_values =
      adjust_slice_indices(input.size(dim), &start, &end, step);

  check_input_args(input, src, dim, num_values, step, out);

  size_t dim_length = input.size(dim);
  size_t leading_dims = getLeadingDims(input, dim);
  size_t trailing_dims = getTrailingDims(input, dim);

  // To start, copy the input into the output
  memcpy(out.mutable_data_ptr(), input.const_data_ptr(), input.nbytes());

  ScalarType in_type = input.scalar_type();
  ScalarType src_type = src.scalar_type();

  ET_SWITCH_REAL_TYPES_AND(Bool, in_type, ctx, __func__, CTYPE, [&]() {
    ET_SWITCH_REAL_TYPES_AND(Bool, src_type, ctx, __func__, CTYPE_SRC, [&]() {
      CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
      const CTYPE_SRC* src_data = src.const_data_ptr<CTYPE_SRC>();

      size_t src_offset = 0;

      for (int i = 0; i < leading_dims; i++) {
        size_t out_offset = (i * dim_length + start) * trailing_dims;
        for (int j = 0; j < num_values; j++) {
          for (size_t k = 0; k < trailing_dims; ++k) {
            out_data[out_offset + k] =
                convert<CTYPE, CTYPE_SRC>(src_data[src_offset + k]);
          }
          src_offset += trailing_dims;
          out_offset += step * trailing_dims;
        }
      }
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
