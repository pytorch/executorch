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

// CUDA error checking macro
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

// Kernel launch check macro
#define ET_CUDA_KERNEL_LAUNCH_CHECK_OR_RETURN_ERROR() \
  ET_CUDA_CHECK_OR_RETURN_ERROR(cudaGetLastError())

namespace executorch::backends::cuda {

// Enum for supported data types in et-cuda backend
enum class SupportedDTypes : int32_t {
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
using executorch::runtime::Error;
// Common AOTI type aliases
using AOTITorchError = Error;

// Helper function to check if a dtype is supported in ET CUDA backend
inline bool is_dtype_supported_in_et_cuda(int32_t dtype) {
  switch (dtype) {
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
      "Unsupported dtype: %d. Supported dtypes: %d (int64), %d (float32), %d (bfloat16)",
      dtype,
      static_cast<int32_t>(SupportedDTypes::INT64),
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDTypes::BFLOAT16));

  return Error::Ok;
}
} // extern "C"

} // namespace executorch::backends::cuda
