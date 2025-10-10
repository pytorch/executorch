/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/aoti/utils.h>
#include <executorch/backends/apple/metal/runtime/shims/types.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <cstdint>

namespace executorch {
namespace backends {
namespace metal {

// Enum for supported data types in et-metal backend
enum class SupportedDTypes : int32_t {
  // UINT8 = 0,    // PyTorch's uint8 dtype code
  // INT8 = 1,     // PyTorch's int8 dtype code
  // INT16 = 2,    // PyTorch's int16 dtype code
  // INT32 = 3,    // PyTorch's int32 dtype code
  INT64 = 4, // PyTorch's int64 dtype code
  // FLOAT16 = 5,  // PyTorch's float16 dtype code
  FLOAT32 = 6, // PyTorch's float32 dtype code
  // FLOAT64 = 7,  // PyTorch's float64 dtype code
  // BOOL = 11,    // PyTorch's bool dtype code
  BFLOAT16 = 15 // PyTorch's bfloat16 dtype code
};

extern "C" {

// Helper function to check if a dtype is supported in Metal backend
bool is_dtype_supported_in_et_metal(int32_t dtype);

// Metal-specific dtype validation utility function
AOTITorchError validate_dtype(int32_t dtype);

} // extern "C"

// Utility function to convert sizes pointer to vector
std::vector<executorch::aten::SizesType> convert_sizes_to_vector(
    int64_t ndim,
    const int64_t* sizes_ptr);

// Utility function to convert strides pointer to vector or calculate from sizes
std::vector<executorch::aten::StridesType> convert_strides_to_vector(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr);

// Check if tensor is in contiguous memory format (NCHW for 4D tensors)
// Contiguous format means strides decrease from left to right:
// For NCHW: strides = [C*H*W, H*W, W, 1]
inline bool is_contiguous_tensor(
    std::vector<executorch::aten::SizesType> sizes,
    std::vector<executorch::aten::StridesType> strides) {
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

} // namespace metal
} // namespace backends
} // namespace executorch
