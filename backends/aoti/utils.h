/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <cstddef>
#include <cstdint>

namespace executorch {
namespace backends {
namespace aoti {

// Common using declarations for ExecutorTorch types
using executorch::runtime::Error;

extern "C" {

// Common AOTI type aliases
using AOTITorchError = Error;

// Map int32_t dtype to number of bytes per element (reusing ExecutorTorch's
// elementSize function)
size_t dtype_to_element_size(int32_t dtype);

// Map int32_t dtype to ExecutorTorch ScalarType (robust version of hardcoded
// ScalarType::Float)
executorch::aten::ScalarType dtype_to_scalar_type(int32_t dtype);

// Storage offset validation utility function
AOTITorchError validate_storage_offset(int64_t storage_offset);

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch