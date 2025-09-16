/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <cstdint>
#include "types.h"

namespace executorch {
namespace backends {
namespace aoti {

// Enum for supported data types in et-cuda backend
enum class SupportedDTypes : int32_t {
  FLOAT32 = 6, // PyTorch's float32 dtype code

  // BOOL = 11,    // PyTorch's bool dtype code
  // UINT8 = 1,    // PyTorch's uint8 dtype code
  // INT8 = 2,     // PyTorch's int8 dtype code
  // INT16 = 3,    // PyTorch's int16 dtype code
  // INT32 = 4,    // PyTorch's int32 dtype code
  // INT64 = 5,    // PyTorch's int64 dtype code
  // FLOAT16 = 7,  // PyTorch's float16 dtype code
  // FLOAT64 = 8,  // PyTorch's float64 dtype code
  // BFLOAT16 = 15 // PyTorch's bfloat16 dtype code
};

extern "C" {

// Helper function to check if a dtype is supported
bool is_dtype_supported_in_et_cuda(int32_t dtype);

// Map int32_t dtype to number of bytes per element (reusing ExecutorTorch's
// elementSize function)
size_t dtype_to_element_size(int32_t dtype);

// Map int32_t dtype to ExecutorTorch ScalarType (robust version of hardcoded
// ScalarType::Float)
executorch::aten::ScalarType dtype_to_scalar_type(int32_t dtype);

// Cleanup function for tensor output file (called during backend destruction)
void cleanup_aoti_tensor_output();

// Dtype validation utility function
AOTITorchError validate_dtype(int32_t dtype);

// Storage offset validation utility function
AOTITorchError validate_storage_offset(int64_t storage_offset);

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch
