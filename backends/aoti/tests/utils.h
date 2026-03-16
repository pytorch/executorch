/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>
#include <vector>

namespace executorch {
namespace backends {
namespace aoti {
namespace test {

// Use the same type aliases as in common_shims.h
using executorch::runtime::etensor::Tensor;

/**
 * Creates a test tensor with the specified shape and scalar type
 */
inline std::shared_ptr<Tensor> create_test_tensor(
    const std::vector<int64_t>& sizes,
    exec_aten::ScalarType dtype = exec_aten::ScalarType::Float) {
  // Calculate total number of elements
  int64_t total_elements = 1;
  for (int64_t size : sizes) {
    total_elements *= size;
  }

  // Calculate strides (row-major layout)
  std::vector<int64_t> strides(sizes.size());
  if (sizes.size() > 0) {
    strides[sizes.size() - 1] = 1;
    for (int i = sizes.size() - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * sizes[i + 1];
    }
  }

  // Allocate data buffer
  size_t dtype_size = exec_aten::elementSize(dtype);
  void* data = malloc(total_elements * dtype_size);

  // Convert sizes and strides to the required type
  std::vector<executorch::aten::SizesType> sizes_converted(
      sizes.begin(), sizes.end());
  std::vector<executorch::aten::SizesType> strides_converted(
      strides.begin(), strides.end());

  // Create the tensor with the correct argument types and count
  auto tensor = executorch::extension::from_blob(
      data, sizes_converted, strides_converted, dtype);

  return tensor;
}

/**
 * Helper to clean up tensor data that was allocated with malloc
 */
inline void free_tensor_data(Tensor* tensor) {
  if (tensor && tensor->mutable_data_ptr()) {
    free(tensor->mutable_data_ptr());
  }
}

} // namespace test
} // namespace aoti
} // namespace backends
} // namespace executorch
