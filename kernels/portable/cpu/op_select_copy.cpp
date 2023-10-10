/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

Tensor& select_copy_int_out(
    RuntimeContext& ctx,
    const Tensor& in,
    int64_t dim,
    int64_t index,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_select_copy_out_args(in, dim, index, out),
      InvalidArgument,
      out);

  if (dim < 0) {
    dim += nonzero_dim(in);
  }

  Tensor::SizesType target_sizes[kTensorDimensionLimit];
  size_t target_ndim = 0;
  get_select_copy_out_target_size(in, dim, target_sizes, &target_ndim);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {target_sizes, target_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  // If the input is a empty tensor, no other operation could be done. We just
  // return the output.
  if (in.numel() == 0) {
    return out;
  }
  // The code past this point assumes that the tensors are non-empty.

  // Support python-style negative indexing
  if (index < 0) {
    index += in.size(dim);
  }

  size_t leading_dims = getLeadingDims(in, dim);
  size_t trailing_dims = getTrailingDims(in, dim);
  size_t dim_length = in.size(dim);

  // Number of bytes to copy in the each memcpy operation
  size_t copy_size_per_op = trailing_dims * out.element_size();

  // Step between the src locations of two adjcant memcpy operations
  size_t src_step_per_op = dim_length * trailing_dims * in.element_size();

  // the start point of data need to be copied is the start point of overall
  // data chunk plus the offset between the overall start point and the first
  // data to be copied.
  char* input_data = in.mutable_data_ptr<char>();

  size_t start_offset = index * trailing_dims * in.element_size();
  char* src = input_data + start_offset;

  char* dest = out.mutable_data_ptr<char>();

  for (size_t j = 0; j < leading_dims; ++j) {
    memcpy(dest, src, copy_size_per_op);
    src += src_step_per_op;
    dest += copy_size_per_op;
  }
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
