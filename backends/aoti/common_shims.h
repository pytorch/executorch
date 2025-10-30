/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

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

extern "C" {

// Common AOTI type aliases
using AOTIRuntimeError = Error;
using AOTITorchError = Error;

// Global storage for tensor metadata
extern std::unordered_map<Tensor*, std::vector<int64_t>> tensor_to_sizes;
extern std::unordered_map<Tensor*, std::vector<int64_t>> tensor_to_strides;

// Attribute-related operations (memory-irrelevant)
AOTITorchError aoti_torch_get_data_ptr(Tensor* tensor, void** ret_data_ptr);

AOTITorchError aoti_torch_get_storage_offset(
    Tensor* tensor,
    int64_t* ret_storage_offset);

AOTITorchError aoti_torch_get_strides(Tensor* tensor, int64_t** ret_strides);

AOTITorchError aoti_torch_get_dtype(Tensor* tensor, int32_t* ret_dtype);

AOTITorchError aoti_torch_get_sizes(Tensor* tensor, int64_t** ret_sizes);

AOTITorchError aoti_torch_get_storage_size(Tensor* tensor, int64_t* ret_size);

AOTITorchError aoti_torch_get_device_index(
    Tensor* tensor,
    int32_t* ret_device_index);

AOTITorchError aoti_torch_get_dim(Tensor* tensor, int64_t* ret_dim);

// Utility functions for device and layout information
int32_t aoti_torch_device_type_cpu();
int32_t aoti_torch_layout_strided();
int32_t aoti_torch_dtype_float32();
int32_t aoti_torch_dtype_bfloat16();
int32_t aoti_torch_dtype_int8();
int32_t aoti_torch_dtype_int16();
int32_t aoti_torch_dtype_int32();
int32_t aoti_torch_dtype_int64();
int32_t aoti_torch_dtype_bool();

// Dtype utility function needed by Metal backend
size_t aoti_torch_dtype_element_size(int32_t dtype);

// Autograd mode functions
int32_t aoti_torch_grad_mode_is_enabled();
void aoti_torch_grad_mode_set_enabled(bool enabled);

// Cleanup functions for clearing global state
void cleanup_tensor_metadata();

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch
