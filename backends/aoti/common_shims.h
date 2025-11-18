/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/aoti/export.h>
#include <executorch/backends/aoti/utils.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace executorch {
namespace backends {
namespace aoti {

// Common using declarations for ExecuTorch types
using executorch::runtime::Error;
using executorch::runtime::etensor::Tensor;

// Global storage for tensor metadata
extern std::unordered_map<Tensor*, std::vector<int64_t>> tensor_to_sizes;
extern std::unordered_map<Tensor*, std::vector<int64_t>> tensor_to_strides;

extern "C" {

// Common AOTI type aliases
using AOTIRuntimeError = Error;
using AOTITorchError = Error;

// Attribute-related operations (memory-irrelevant)
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_get_data_ptr(Tensor* tensor, void** ret_data_ptr);

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_get_storage_offset(Tensor* tensor, int64_t* ret_storage_offset);

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_get_strides(Tensor* tensor, int64_t** ret_strides);

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_get_dtype(Tensor* tensor, int32_t* ret_dtype);

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_get_sizes(Tensor* tensor, int64_t** ret_sizes);

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_get_storage_size(Tensor* tensor, int64_t* ret_size);

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_get_device_index(Tensor* tensor, int32_t* ret_device_index);

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_get_dim(Tensor* tensor, int64_t* ret_dim);

// Utility functions for device and layout information
AOTI_SHIM_EXPORT int32_t aoti_torch_device_type_cpu();
AOTI_SHIM_EXPORT int32_t aoti_torch_layout_strided();
AOTI_SHIM_EXPORT int32_t aoti_torch_dtype_float32();
AOTI_SHIM_EXPORT int32_t aoti_torch_dtype_bfloat16();
AOTI_SHIM_EXPORT int32_t aoti_torch_dtype_int8();
AOTI_SHIM_EXPORT int32_t aoti_torch_dtype_int16();
AOTI_SHIM_EXPORT int32_t aoti_torch_dtype_int32();
AOTI_SHIM_EXPORT int32_t aoti_torch_dtype_int64();

// Dtype utility function needed by Metal backend
AOTI_SHIM_EXPORT size_t aoti_torch_dtype_element_size(int32_t dtype);

// Autograd mode functions
AOTI_SHIM_EXPORT int32_t aoti_torch_grad_mode_is_enabled();
AOTI_SHIM_EXPORT void aoti_torch_grad_mode_set_enabled(bool enabled);

// Cleanup functions for clearing global state
AOTI_SHIM_EXPORT void cleanup_tensor_metadata();

AOTI_SHIM_EXPORT void aoti_torch_warn(
    const char* func,
    const char* file,
    uint32_t line,
    const char* msg);

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_get_storage_size(Tensor* tensor, int64_t* ret_size);

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_clone_preserve_strides(Tensor* self, Tensor** ret_new_tensor);

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_clone(Tensor* self, Tensor** ret_new_tensor);

AOTI_SHIM_EXPORT AOTITorchError aoti_torch_create_tensor_from_blob(
    void* data_ptr,
    int64_t ndim,
    const int64_t* sizes,
    const int64_t* strides,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    Tensor** ret_new_tensor);

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch
