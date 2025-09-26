/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <cstdint>
#include <vector>

namespace executorch {
namespace backends {
namespace cuda {

// Enum for supported data types in et-cuda backend
enum class SupportedDTypes : int32_t {
  FLOAT32 = 6, // PyTorch's float32 dtype code
  BFLOAT16 = 15, // PyTorch's bfloat16 dtype code
};

// Enum for supported device types in et-cuda backend
enum class SupportedDevices : int32_t {
  CPU = 0, // CPU device
  CUDA = 1, // CUDA device
};

// Utility function to convert sizes pointer to vector
inline std::vector<executorch::aten::SizesType> convert_sizes_to_vector(
    int64_t ndim,
    const int64_t* sizes_ptr) {
  std::vector<executorch::aten::SizesType> sizes(ndim);
  for (int i = 0; i < ndim; i++) {
    sizes[i] = static_cast<executorch::aten::SizesType>(sizes_ptr[i]);
  }
  return sizes;
}

// Utility function to convert strides pointer to vector or calculate from sizes
inline std::vector<executorch::aten::StridesType> convert_strides_to_vector(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr) {
  std::vector<executorch::aten::StridesType> strides(ndim);

  if (strides_ptr != nullptr) {
    // Use provided strides. it is ok if provided strides here is not contiguous
    // strides since it will be used internally in CUDA delegate.
    for (int64_t i = 0; i < ndim; i++) {
      strides[i] = static_cast<executorch::aten::StridesType>(strides_ptr[i]);
    }
  } else {
    // Calculate strides from sizes using ExecutorTorch's algorithm
    if (ndim > 0) {
      strides[ndim - 1] = static_cast<executorch::aten::StridesType>(
          1); // Last dimension has stride 1
      for (int64_t i = ndim - 2; i >= 0; i--) {
        if (sizes_ptr[i + 1] == 0) {
          strides[i] = strides[i + 1]; // Copy stride when size is 0
        } else {
          strides[i] = static_cast<executorch::aten::StridesType>(
              static_cast<int64_t>(strides[i + 1]) * sizes_ptr[i + 1]);
        }
      }
    }
  }
  return strides;
}

extern "C" {
using executorch::runtime::Error;
// Common AOTI type aliases
using AOTITorchError = Error;

// Helper function to check if a dtype is supported in ET CUDA backend
inline bool is_dtype_supported_in_et_cuda(int32_t dtype) {
  switch (dtype) {
    case static_cast<int32_t>(SupportedDTypes::FLOAT32):
    case static_cast<int32_t>(SupportedDTypes::BFLOAT16):
      return true;
    default:
      return false;
  }
}

// Dtype validation utility function
inline AOTITorchError validate_dtype(int32_t dtype) {
  if (is_dtype_supported_in_et_cuda(dtype)) {
    return Error::Ok;
  }

  ET_LOG(
      Error,
      "Unsupported dtype: %d. Supported dtypes: %d (float32), %d (bfloat16)",
      dtype,
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDTypes::BFLOAT16));
  return Error::InvalidArgument;
}
} // extern "C"

} // namespace cuda
} // namespace backends
} // namespace executorch
