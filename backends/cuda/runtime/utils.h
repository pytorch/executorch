
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
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>
#include <cstddef>
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

namespace executorch {
namespace backends {
namespace cuda {

// Common using declarations for ExecuTorch types
using executorch::runtime::Error;

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

extern "C" {

// Common AOTI type aliases
// AOTITorchError is int32_t to match AOTI shim layer convention
using AOTITorchError = int32_t;

// Storage offset validation utility function
inline AOTITorchError validate_storage_offset(int64_t storage_offset) {

// Map int32_t dtype to ExecuTorch ScalarType (robust version of hardcoded
// ScalarType::Float)
inline executorch::aten::ScalarType dtype_to_scalar_type(int32_t dtype) {
  // Convert based on known PyTorch dtype codes (without CUDA-specific
  // dependency)
  switch (dtype) {
    case 1: // PyTorch's int8 dtype code
      return executorch::aten::ScalarType::Char;
    case 2: // PyTorch's int16 dtype code
      return executorch::aten::ScalarType::Short;
    case 3: // PyTorch's int32 dtype code
      return executorch::aten::ScalarType::Int;
    case 4: // PyTorch's int64 dtype code
      return executorch::aten::ScalarType::Long;
    case 6: // PyTorch's float32 dtype code
      return executorch::aten::ScalarType::Float;
    case 11: // PyTorch's bool dtype code
      return executorch::aten::ScalarType::Bool;
    case 15: // PyTorch's bfloat16 dtype code
      return executorch::aten::ScalarType::BFloat16;
    // Future support for additional dtypes can be added here
    default:
      ET_LOG(Error, "Unsupported dtype: %d for ScalarType conversion", dtype);
      return executorch::aten::ScalarType::Undefined;
  }
}

// Map int32_t dtype to number of bytes per element (reusing ExecuTorch's
// elementSize function)
inline size_t dtype_to_element_size(int32_t dtype) {
  // First convert int32_t dtype to ExecuTorch ScalarType, then use existing
  // elementSize function
  executorch::aten::ScalarType scalar_type = dtype_to_scalar_type(dtype);
  if (scalar_type == executorch::aten::ScalarType::Undefined) {
    ET_LOG(Error, "Unsupported dtype: %d for element size calculation", dtype);
    return 0; // Return 0 to indicate error
  }

  // Reuse ExecuTorch's existing elementSize function from scalar_type_util.h
  return executorch::runtime::elementSize(scalar_type);
}

// Storage offset validation utility function
inline AOTITorchError validate_storage_offset(int64_t storage_offset) {
  // Storage offset must always be 0
  if (storage_offset != 0) {
    ET_LOG(
        Error,
        "Storage offset must be 0. Got storage_offset: %ld",
        storage_offset);
    return Error::InvalidArgument;
  }
  return Error::Ok;
}

// Check if tensor is in contiguous memory format (NCHW for 4D tensors)
// Contiguous format means strides decrease from left to right:
// For NCHW: strides = [C*H*W, H*W, W, 1]
inline bool is_tensor_contiguous(
    int64_t ndim,
    const int64_t* sizes,
    const int64_t* strides) {
  int64_t expected_stride = 1;
  for (int64_t i = ndim - 1; i >= 0; i--) {
    if (strides[i] != expected_stride) {
      return false;
    }
    expected_stride *= sizes[i];
  }
  return true;
}

} // extern "C"

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
    // Use provided strides.
    for (int64_t i = 0; i < ndim; i++) {
      strides[i] = static_cast<executorch::aten::StridesType>(strides_ptr[i]);
    }
  } else {
    // Calculate strides from sizes.
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

// Check if tensor is in contiguous memory format (NCHW for 4D tensors)
// Contiguous format means strides decrease from left to right:
// For NCHW: strides = [C*H*W, H*W, W, 1]
inline bool is_contiguous_tensor(
    std::vector<executorch::aten::SizesType>& sizes,
    std::vector<executorch::aten::StridesType>& strides) {
  int64_t ndim = static_cast<int64_t>(strides.size());
  int64_t expected_stride = 1;
  for (int64_t i = ndim - 1; i >= 0; i--) {
    if (strides[i] != expected_stride) {
      return false;
    }
    expected_stride *= sizes[i];
  }
  return true;
}

} // namespace cuda
} // namespace backends
} // namespace executorch
