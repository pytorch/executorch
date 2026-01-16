/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/aoti/export.h>
#include <executorch/backends/aoti/slim/core/SlimTensor.h>
#include <executorch/runtime/core/error.h>
#include <cstdint>

namespace executorch {
namespace backends {
namespace aoti {

// Common using declarations for ExecuTorch types
using executorch::runtime::Error;

// Tensor type definition using SlimTensor
using Tensor = executorch::backends::aoti::slim::SlimTensor;

// Common AOTI type aliases
using AOTIRuntimeError = Error;
using AOTITorchError = Error;

// ============================================================
// Basic Property Getters - Declarations
// ============================================================

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_get_data_ptr(Tensor* tensor, void** ret_data_ptr);

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_get_sizes(Tensor* tensor, int64_t** ret_sizes);

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_get_strides(Tensor* tensor, int64_t** ret_strides);

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_get_dtype(Tensor* tensor, int32_t* ret_dtype);

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_get_dim(Tensor* tensor, int64_t* ret_dim);

// ============================================================
// Storage & Device Property Getters - Declarations
// ============================================================

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_get_storage_offset(Tensor* tensor, int64_t* ret_storage_offset);

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_get_storage_size(Tensor* tensor, int64_t* ret_size);

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_get_device_type(Tensor* tensor, int32_t* ret_device_type);

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_get_device_index(Tensor* tensor, int32_t* ret_device_index);

} // namespace aoti
} // namespace backends
} // namespace executorch
