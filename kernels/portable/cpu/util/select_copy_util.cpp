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
#include "executorch/kernels/portable/cpu/util/select_copy_util.h"

namespace torch {
namespace executor {

using Tensor = exec_aten::Tensor;

Error select_copy_util(
    const Tensor& in,
    int64_t dim,
    int64_t index,
    Tensor& out) {
  if (!check_select_copy_out_args(in, dim, index, out)) {
    return Error::InvalidArgument;
  }

  if (dim < 0) {
    dim += nonzero_dim(in);
  }

  Tensor::SizesType target_sizes[kTensorDimensionLimit];
  size_t target_ndim = 0;
  get_select_copy_out_target_size(in, dim, target_sizes, &target_ndim);

  if (!(resize_tensor(out, {target_sizes, target_ndim}) == Error::Ok)) {
    return Error::InvalidArgument;
  }

  if (!tensors_have_same_dim_order(in, out)) {
    return Error::InvalidArgument;
  }

  // If the input is a empty tensor, no other operation could be done. We just
  // return the output.
  if (in.numel() == 0) {
    return Error::Ok;
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

  return Error::Ok;
}

} // namespace executor
} // namespace torch
