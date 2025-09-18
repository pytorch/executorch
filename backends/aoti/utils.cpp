/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "utils.h"
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace backends {
namespace aoti {

extern "C" {

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
  // Convert based on known PyTorch dtype codes (without CUDA-specific
  // dependency)
  switch (dtype) {
    case 6: // PyTorch's float32 dtype code
      return executorch::aten::ScalarType::Float;
    // Future support for additional dtypes can be added here
    // case 11:    // PyTorch's bool dtype code
    //   return executorch::aten::ScalarType::Bool;
    // case 1:     // PyTorch's uint8 dtype code
    //   return executorch::aten::ScalarType::Byte;
    // case 2:     // PyTorch's int8 dtype code
    //   return executorch::aten::ScalarType::Char;
    // case 3:     // PyTorch's int16 dtype code
    //   return executorch::aten::ScalarType::Short;
    // case 4:     // PyTorch's int32 dtype code
    //   return executorch::aten::ScalarType::Int;
    // case 5:     // PyTorch's int64 dtype code
    //   return executorch::aten::ScalarType::Long;
    // case 7:     // PyTorch's float16 dtype code
    //   return executorch::aten::ScalarType::Half;
    // case 8:     // PyTorch's float64 dtype code
    //   return executorch::aten::ScalarType::Double;
    // case 15:    // PyTorch's bfloat16 dtype code
    //   return executorch::aten::ScalarType::BFloat16;
    default:
      ET_LOG(Error, "Unsupported dtype: %d for ScalarType conversion", dtype);
      return executorch::aten::ScalarType::Undefined;
  }
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