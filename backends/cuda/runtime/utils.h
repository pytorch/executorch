/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/error.h>
#include <cstdint>

namespace executorch {
namespace backends {
namespace aoti {

// Common using declarations for ExecutorTorch types
using executorch::runtime::Error;

extern "C" {

// Common AOTI type aliases
using AOTITorchError = Error;

// Helper function to check if a dtype is supported in ET CUDA backend
bool is_dtype_supported_in_et_cuda(int32_t dtype);

// Dtype validation utility function
AOTITorchError validate_dtype(int32_t dtype);

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch
