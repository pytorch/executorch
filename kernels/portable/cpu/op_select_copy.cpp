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
void check_and_update_select_copy_int_out_args(
    const Tensor input,
    int64_t dim,
    int64_t index,
    Tensor output) {
  // Support python-style negative indexing. E.g., for the shape {2, 3, 4},
  // dim = -1 would refer to dim[2], dim = -2 would refer to dim[1], and so on.

  // The dim planed to be selected on shall exist in input
  ET_CHECK_MSG(
      dim >= -input.dim() && dim < input.dim(),
      "dim %" PRId64 " out of range [-%zd,%zd)",
      dim,
      input.dim(),
      input.dim());

  // The index shall be valid in the given dimenson
  ET_CHECK_MSG(
      index >= -input.size(dim) && index < input.size(dim),
      "index %" PRId64 " out of range [-%zd,%zd) at input.size( %" PRId64 ")",
      index,
      input.size(dim),
      input.size(dim),
      dim);

  // Support python-style negative indexing
  if (dim < 0) {
    dim += input.dim();
  }
  if (index < 0) {
    index += input.size(dim);
  }

  // Input dtype shall match the output dtype.
  ET_CHECK_SAME_DTYPE2(input, output);

  // The output.dim() shall be one lower than input.dim() since we create output
  // by selecting data on one dim of input
  // https://pytorch.org/docs/stable/generated/torch.select.html
  ET_CHECK_MSG(
      input.dim() == output.dim() + 1,
      "input.dim() %zd != output.dim() + 1 %zd",
      input.dim(),
      output.dim() + 1);

  // The size of output tensor should follow these rules:
  // - output.size(i) shall equal to input.size(i) if i < dim,
  // - output.size(i) shall equal to input.size(i+1) if i >= dim

  for (size_t d = 0; d < input.dim() - 1; d++) {
    if (d < dim) {
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
          input.size(d + 1) == output.size(d),
          "input.size(%zu) %zd != output.size(%zu) %zd | dim = %" PRId64 ")",
          d + 1,
          input.size(d + 1),
          d,
          output.size(d),
          dim);
    }
  }
}
} // namespace

/// select_copy.int_out(Tensor self, int dim, int index, *, Tensor(a!) output)
/// -> Tensor(a!)
Tensor& select_copy_int_out(
    RuntimeContext& ctx,
    const Tensor& input,
    int64_t dim,
    int64_t index,
    Tensor& output) {
  (void)ctx;
  // Assert that the args are valid.
  check_and_update_select_copy_int_out_args(input, dim, index, output);

  // If the input is a empty tensor, no other operation could be done. We just
  // return the output.
  if (input.numel() == 0) {
    return output;
  }
  // The code past this point assumes that the tensors are non-empty.

  // Support python-style negative indexing
  if (dim < 0) {
    dim += input.dim();
  }
  if (index < 0) {
    index += input.size(dim);
  }

  size_t leading_dims = getLeadingDims(input, dim);
  size_t trailing_dims = getTrailingDims(input, dim);
  size_t dim_length = input.size(dim);

  // Number of bytes to copy in the each memcpy operation
  size_t copy_size_per_op = trailing_dims * output.element_size();

  // Step between the src locations of two adjcant memcpy operations
  size_t src_step_per_op = dim_length * trailing_dims * input.element_size();

  // the start point of data need to be copied is the start point of overall
  // data chunk plus the offset between the overall start point and the first
  // data to be copied.
  char* input_data = input.mutable_data_ptr<char>();

  size_t start_offset = index * trailing_dims * input.element_size();
  char* src = input_data + start_offset;

  char* dest = output.mutable_data_ptr<char>();

  for (size_t j = 0; j < leading_dims; ++j) {
    memcpy(dest, src, copy_size_per_op);
    src += src_step_per_op;
    dest += copy_size_per_op;
  }
  return output;
}

} // namespace native
} // namespace executor
} // namespace torch
