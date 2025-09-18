/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "utils.h"
#include <executorch/runtime/platform/log.h>

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

// Helper function to check if a dtype is supported in ET CUDA backend
bool is_dtype_supported_in_et_cuda(int32_t dtype) {
  switch (dtype) {
    case static_cast<int32_t>(SupportedDTypes::FLOAT32):
      return true;
    // case static_cast<int32_t>(SupportedDTypes::BOOL):
    // case static_cast<int32_t>(SupportedDTypes::UINT8):
    // case static_cast<int32_t>(SupportedDTypes::INT8):
    // case static_cast<int32_t>(SupportedDTypes::INT16):
    // case static_cast<int32_t>(SupportedDTypes::INT32):
    // case static_cast<int32_t>(SupportedDTypes::INT64):
    // case static_cast<int32_t>(SupportedDTypes::FLOAT16):
    // case static_cast<int32_t>(SupportedDTypes::FLOAT64):
    // case static_cast<int32_t>(SupportedDTypes::BFLOAT16):
    //   return true;
    default:
      return false;
  }
}

// Dtype validation utility function
AOTITorchError validate_dtype(int32_t dtype) {
  if (is_dtype_supported_in_et_cuda(dtype)) {
    return Error::Ok;
  }

  ET_LOG(
      Error,
      "Unsupported dtype: %d. Supported dtypes: %d (float32)",
      dtype,
      static_cast<int32_t>(SupportedDTypes::FLOAT32));
  return Error::InvalidArgument;
}

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch
