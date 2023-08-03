/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cinttypes>
#include <cstdint>
#include <cstring>

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

namespace {

/**
 * Assumptions for inputs:
 * 1. output size is the same as input size
 * 2. src size is the same as the selected slice from the input
 * 3. dim and index values are valid given the input tensor
 */
void check_select_scatter_args(
    const Tensor& input,
    const Tensor& src,
    int64_t dim,
    int64_t index,
    Tensor& output) {
  // Support python-style negative indexing. E.g., for the shape {2, 3, 4},
  // dim = -1 would refer to dim[2], dim = -2 would refer to dim[1], and so on.

  // The dim planed to be selected on shall exist in input
  ET_CHECK_MSG(
      dim >= 0 && dim < input.dim(),
      "dim %" PRId64 " out of range [-%zd,%zd)",
      dim,
      input.dim(),
      input.dim());

  // The index shall be valid in the given dimenson
  ET_CHECK_MSG(
      index >= 0 && index < input.size(dim),
      "index %" PRId64 " out of range [-%zd,%zd) at input.size( %" PRId64 ")",
      index,
      input.size(dim),
      input.size(dim),
      dim);

  // All tensors should be same dtype
  ET_CHECK_SAME_DTYPE3(input, output, src);

  // The size of output tensor should be the same as the input
  ET_CHECK_SAME_SHAPE2(input, output);

  // The src.dim() shall be one lower than input.dim() since src needs to fit
  // into the selected data on one dim of input
  // https://pytorch.org/docs/stable/generated/torch.select_scatter.html
  ET_CHECK_MSG(
      input.dim() == src.dim() + 1,
      "input.dim() %zd != src.dim() + 1 %zd",
      input.dim(),
      src.dim() + 1);

  // The size of src tensor should follow these rules:
  // - src.size(i) shall equal to input.size(i) if i < dim,
  // - src.size(i) shall equal to input.size(i+1) if i >= dim

  for (ssize_t d = 0; d < input.dim() - 1; d++) {
    if (d < dim) {
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
          input.size(d + 1) == src.size(d),
          "input.size(%zu) %zd != src.size(%zu) %zd | dim = %" PRId64 ")",
          d + 1,
          input.size(d + 1),
          d,
          src.size(d),
          dim);
    }
  }
}

} // namespace

/// aten::select_scatter.out(Tensor self, Tensor src, int dim, SymInt index, *,
/// Tensor(a!) out) -> Tensor(a!)
Tensor& select_scatter_out(
    RuntimeContext& context,
    const Tensor& input,
    const Tensor& src,
    int64_t dim,
    int64_t index,
    Tensor& out) {
  // Avoid unused variable warning
  (void)context;

  // Account for negative indices
  if (dim < 0) {
    dim += input.dim();
  }
  if (index < 0) {
    index += input.size(dim);
  }

  // Resize the tensor to the expected output size
  Tensor::SizesType expected_output_size[16];
  for (size_t i = 0; i < input.dim(); ++i) {
    expected_output_size[i] = input.size(i);
  }
  auto error = resize_tensor(
      out, {expected_output_size, static_cast<size_t>(input.dim())});
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");

  // Check args
  check_select_scatter_args(input, src, dim, index, out);

  // If the input is a empty tensor, no other operation could be done. We just
  // return the output.
  if (input.numel() == 0) {
    return out;
  }

  // To start, copy the input into the output. Input will not be empty due to
  // the checks performed above.
  memcpy(out.mutable_data_ptr(), input.const_data_ptr(), input.nbytes());

  // Strides to help with memory address arithmetic
  size_t leading_dims = getLeadingDims(input, dim);
  size_t trailing_stride = getTrailingDims(input, dim);

  size_t dim_length = input.size(dim);

  // Number of bytes to copy for each memcpy
  size_t copy_nbytes = trailing_stride * src.element_size();

  // Number of bytes to step forward to reach the next copy output location
  size_t out_step_nbytes = dim_length * trailing_stride * out.element_size();

  // Position data pointers at the starting point
  size_t start_offset = index * trailing_stride * out.element_size();
  char* out_data = out.mutable_data_ptr<char>() + start_offset;

  const char* src_data = src.const_data_ptr<char>();

  for (size_t step = 0; step < leading_dims; ++step) {
    memcpy(out_data, src_data, copy_nbytes);
    out_data += out_step_nbytes;
    src_data += copy_nbytes;
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
