/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "executorch/kernels/portable/cpu/util/sort_util.h"
#include <executorch/runtime/kernel/kernel_includes.h>
#include <algorithm>

namespace torch {
namespace executor {

using Tensor = exec_aten::Tensor;

Error sort_tensor(
    const Tensor& tensor,
    Tensor& sorted_tensor,
    Tensor& sorted_indices,
    bool descending) {
  // Check if the input tensor is a valid input
  ET_CHECK_MSG(tensor.dim() == 1, "Input tensor must be 1D");

  // Check if the output tensors are valid
  ET_CHECK_MSG(sorted_tensor.dim() == 1, "Output tensor must be 1D");
  ET_CHECK_MSG(sorted_indices.dim() == 1, "Output tensor must be 1D");

  // Check if the output tensors have the same dtype
  ET_CHECK_MSG(
      tensor.scalar_type() == sorted_tensor.scalar_type(),
      "Input and output tensors must have the same dtype");
  ET_CHECK_MSG(
      tensor.scalar_type() == ScalarType::Float,
      "Only float inputs are supported currently");
  ET_CHECK_MSG(
      sorted_indices.scalar_type() == exec_aten::ScalarType::Long,
      "Output tensor must be of type int64");

  // Get the number of elements in the tensor
  int size = tensor.numel();

  // Create a tensor to store the indices
  for (int i = 0; i < size; i++) {
    sorted_indices.mutable_data_ptr<int64_t>()[i] = i;
  }

  // Sort the indices based on the corresponding tensor values
  std::sort(
      sorted_indices.mutable_data_ptr<int64_t>(),
      sorted_indices.mutable_data_ptr<int64_t>() + size,
      [&tensor, descending](int64_t i, int64_t j) {
        if (descending) {
          return tensor.const_data_ptr<float>()[i] >
              tensor.const_data_ptr<float>()[j];
        } else {
          return tensor.const_data_ptr<float>()[i] <
              tensor.const_data_ptr<float>()[j];
        }
      });

  // Rearrange the tensor values based on the sorted indices
  for (int i = 0; i < size; i++) {
    sorted_tensor.mutable_data_ptr<float>()[i] = tensor.const_data_ptr<
        float>()[sorted_indices.const_data_ptr<int64_t>()[i]];
  }

  return Error::Ok;
}

} // namespace executor
} // namespace torch
