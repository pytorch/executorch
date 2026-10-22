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
    case static_cast<int32_t>(SupportedDTypes::INT32):
    case static_cast<int32_t>(SupportedDTypes::INT64):
    case static_cast<int32_t>(SupportedDTypes::FLOAT32):
    case static_cast<int32_t>(SupportedDTypes::BOOL):
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

  ET_LOG(Error, "Unsupported dtype: %d", dtype);
  return Error::InvalidArgument;
}

} // extern "C"

bool is_row_major_dense(const Tensor& tensor) {
  const auto sizes = tensor.sizes();
  const auto strides = tensor.strides();
  int64_t expected = 1;
  for (int64_t d = tensor.dim() - 1; d >= 0; d--) {
    if (sizes[d] == 1) {
      continue;
    }
    if (strides[d] != expected) {
      return false;
    }
    expected *= sizes[d];
  }
  return true;
}

} // namespace metal
} // namespace backends
} // namespace executorch
