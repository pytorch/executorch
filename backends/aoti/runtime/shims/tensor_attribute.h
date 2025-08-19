/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/error.h>
#include <unordered_map>
#include <vector>

namespace executorch {
namespace backends {
namespace aoti {

using executorch::runtime::Error;
using executorch::runtime::etensor::Tensor;

extern "C" {

// Type definitions
using AOTITensorHandle = Tensor*;
using AOTIRuntimeError = Error;
using AOTITorchError = Error;

// Global storage for tensor metadata
extern std::unordered_map<Tensor*, std::vector<int64_t>> tensor_to_sizes;
extern std::unordered_map<Tensor*, std::vector<int64_t>> tensor_to_strides;

// Attribute-related operations (memory-irrelevant)
AOTITorchError aoti_torch_get_data_ptr(
    AOTITensorHandle tensor,
    void** ret_data_ptr);

AOTITorchError aoti_torch_get_storage_offset(
    AOTITensorHandle tensor,
    int64_t* ret_storage_offset);

AOTITorchError aoti_torch_get_strides(
    AOTITensorHandle tensor,
    int64_t** ret_strides);

AOTITorchError aoti_torch_get_dtype(
    AOTITensorHandle tensor,
    int32_t* ret_dtype);

AOTITorchError aoti_torch_get_sizes(
    AOTITensorHandle tensor,
    int64_t** ret_sizes);

AOTITorchError aoti_torch_get_storage_size(
    AOTITensorHandle tensor,
    int64_t* ret_size);

// Utility functions for device and layout information
int32_t aoti_torch_device_type_cpu();
int32_t aoti_torch_device_type_cuda();
int32_t aoti_torch_layout_strided();
int32_t aoti_torch_dtype_float32();

// Autograd mode functions
int32_t aoti_torch_grad_mode_is_enabled();
void aoti_torch_grad_mode_set_enabled(bool enabled);

// Cleanup function for clearing global state
void cleanup_tensor_metadata();

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch
