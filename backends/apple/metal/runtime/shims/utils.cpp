/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/apple/metal/runtime/shims/utils.h>
#include <executorch/runtime/platform/log.h>
#include <cstdint>

namespace executorch {
namespace backends {
namespace metal {

extern "C" {

// Helper function to check if a dtype is supported in Metal backend
bool is_dtype_supported_in_et_metal(int32_t dtype) {
  switch (dtype) {
    case static_cast<int32_t>(SupportedDTypes::UINT8):
    case static_cast<int32_t>(SupportedDTypes::INT64):
    case static_cast<int32_t>(SupportedDTypes::FLOAT32):
    case static_cast<int32_t>(SupportedDTypes::BFLOAT16):
      return true;
    default:
      return false;
  }
}

// Metal-specific dtype validation utility function
AOTITorchError validate_dtype(int32_t dtype) {
  if (is_dtype_supported_in_et_metal(dtype)) {
    return Error::Ok;
  }

  ET_LOG(
      Error,
      "Unsupported dtype: %d. Supported dtypes: %d (uint8), %d (int64), %d (float32), %d (bfloat16)",
      dtype,
      static_cast<int32_t>(SupportedDTypes::UINT8),
      static_cast<int32_t>(SupportedDTypes::INT64),
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDTypes::BFLOAT16));
  return Error::InvalidArgument;
}

} // extern "C"

} // namespace metal
} // namespace backends
} // namespace executorch
