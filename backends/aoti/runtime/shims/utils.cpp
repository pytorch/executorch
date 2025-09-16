/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "utils.h"
#include <executorch/runtime/platform/log.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace executorch {
namespace backends {
namespace aoti {

namespace internal {
// Constants for file operations
const char* const TENSOR_OUTPUT_FILENAME =
    "/home/gasoonjia/executorch/aoti_intermediate_output.txt";
} // namespace internal

extern "C" {

// Function to cleanup the tensor output file (to be called from
// aoti_backend.cpp)
void cleanup_aoti_tensor_output() {
  // No cleanup needed since file is opened and closed on each call
}

// Helper function to check if a dtype is supported
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

// Map int32_t dtype to number of bytes per element (reusing ExecutorTorch's
// elementSize function)
size_t dtype_to_element_size(int32_t dtype) {
  // First convert int32_t dtype to ExecutorTorch ScalarType, then use existing
  // elementSize function
  executorch::aten::ScalarType scalar_type = dtype_to_scalar_type(dtype);
  if (scalar_type == executorch::aten::ScalarType::Undefined) {
    ET_LOG(Error, "Unsupported dtype: %d for element size calculation", dtype);
    return 0; // Return 0 to indicate error
  }

  // Reuse ExecutorTorch's existing elementSize function from scalar_type_util.h
  return executorch::runtime::elementSize(scalar_type);
}

// Map int32_t dtype to ExecutorTorch ScalarType (robust version of hardcoded
// ScalarType::Float)
executorch::aten::ScalarType dtype_to_scalar_type(int32_t dtype) {
  // First check if the dtype is supported
  if (!is_dtype_supported_in_et_cuda(dtype)) {
    ET_LOG(Error, "Unsupported dtype: %d for ScalarType conversion", dtype);
    return executorch::aten::ScalarType::Undefined;
  }

  // If supported, use switch to convert
  switch (dtype) {
    case static_cast<int32_t>(SupportedDTypes::FLOAT32):
      return executorch::aten::ScalarType::Float;
    default:
      ET_LOG(
          Error, "Unexpected error in dtype conversion for dtype: %d", dtype);
      return executorch::aten::ScalarType::Undefined;
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

// Storage offset validation utility function
AOTITorchError validate_storage_offset(int64_t storage_offset) {
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

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch
