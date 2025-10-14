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

// CUDA error checking macro (with return)
#define ET_CUDA_CHECK_OR_RETURN_ERROR(EXPR) \
  do {                                      \
    const cudaError_t err = EXPR;           \
    if (err == cudaSuccess) {               \
      break;                                \
    }                                       \
    ET_LOG(                                 \
        Error,                              \
        "%s:%d CUDA error: %s",             \
        __FILE__,                           \
        __LINE__,                           \
        cudaGetErrorString(err));           \
    return Error::Internal;                 \
  } while (0)

// CUDA error checking macro (without return, for use in void functions)
#define ET_CUDA_CHECK(EXPR)                                         \
  do {                                                              \
    const cudaError_t err = EXPR;                                   \
    if (err == cudaSuccess) {                                       \
      break;                                                        \
    }                                                               \
    ET_LOG(                                                         \
        Error,                                                      \
        "%s:%d CUDA error: %s",                                     \
        __FILE__,                                                   \
        __LINE__,                                                   \
        cudaGetErrorString(err));                                   \
    ET_CHECK_MSG(false, "CUDA error: %s", cudaGetErrorString(err)); \
  } while (0)

// Kernel launch check macro (with return)
#define ET_CUDA_KERNEL_LAUNCH_CHECK_OR_RETURN_ERROR() \
  ET_CUDA_CHECK_OR_RETURN_ERROR(cudaGetLastError())

// Kernel launch check macro (without return, for use in void functions)
#define ET_CUDA_KERNEL_LAUNCH_CHECK() ET_CUDA_CHECK(cudaGetLastError())

namespace executorch::backends::cuda {

// Enum for supported data types in et-cuda backend
enum class SupportedDTypes : int32_t {
  INT8 = 1, // PyTorch's int8 dtype code
  INT16 = 2, // PyTorch's int16 dtype code
  INT32 = 3, // PyTorch's int32 dtype code
  INT64 = 4, // PyTorch's int64 dtype code
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
    case static_cast<int32_t>(SupportedDTypes::INT8):
    case static_cast<int32_t>(SupportedDTypes::INT16):
    case static_cast<int32_t>(SupportedDTypes::INT32):
    case static_cast<int32_t>(SupportedDTypes::INT64):
    case static_cast<int32_t>(SupportedDTypes::FLOAT32):
    case static_cast<int32_t>(SupportedDTypes::BFLOAT16):
      return true;
    default:
      return false;
  }
}

// Dtype validation utility function
inline AOTITorchError validate_dtype(int32_t dtype) {
  ET_CHECK_OR_RETURN_ERROR(
      is_dtype_supported_in_et_cuda(dtype),
      InvalidArgument,
      "Unsupported dtype: %d. Supported dtypes: %d (int8), %d (int16), %d (int32), %d (int64), %d (float32), %d (bfloat16)",
      dtype,
      static_cast<int32_t>(SupportedDTypes::INT8),
      static_cast<int32_t>(SupportedDTypes::INT16),
      static_cast<int32_t>(SupportedDTypes::INT32),
      static_cast<int32_t>(SupportedDTypes::INT64),
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDTypes::BFLOAT16));

  return Error::Ok;
}
} // extern "C"

} // namespace executorch::backends::cuda
