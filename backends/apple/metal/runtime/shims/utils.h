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

} // namespace metal
} // namespace backends
} // namespace executorch
