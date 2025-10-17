/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/apple/metal/runtime/shims/types.h>

namespace executorch {
namespace backends {
namespace metal {

struct AOTIMetalKernelFunctionOpaque;
using AOTIMetalKernelFunctionHandle = AOTIMetalKernelFunctionOpaque*;

struct AOTIMetalShaderLibraryOpaque;
using AOTIMetalShaderLibraryHandle = AOTIMetalShaderLibraryOpaque*;

#ifdef __cplusplus
extern "C" {
#endif

// MetalShaderLibrary functions
AOTITorchError aoti_torch_mps_create_shader_library(
    const char* metal_shader_source,
    AOTIMetalShaderLibraryHandle* library_handle);

AOTITorchError aoti_torch_mps_delete_shader_library(
    AOTIMetalShaderLibraryHandle library_handle);

AOTITorchError aoti_torch_mps_get_kernel_function(
    AOTIMetalShaderLibraryHandle library_handle,
    const char* kernel_name,
    AOTIMetalKernelFunctionHandle* function_handle);

// MetalKernelFunction functions
AOTITorchError aoti_torch_mps_start_encoding(
    AOTIMetalKernelFunctionHandle func);

AOTITorchError aoti_torch_mps_set_arg_tensor(
    AOTIMetalKernelFunctionHandle func,
    unsigned idx,
    AOTITensorHandle tensor);

AOTITorchError aoti_torch_mps_set_arg_int(
    AOTIMetalKernelFunctionHandle func,
    unsigned idx,
    int64_t val);

// Pure C dispatch functions - single value versions
AOTITorchError aoti_torch_mps_dispatch_single(
    AOTIMetalKernelFunctionHandle func,
    uint64_t length);

AOTITorchError aoti_torch_mps_dispatch_single_with_group_size(
    AOTIMetalKernelFunctionHandle func,
    uint64_t length,
    uint64_t group_size);

// Pure C dispatch functions - array versions
AOTITorchError aoti_torch_mps_dispatch_array(
    AOTIMetalKernelFunctionHandle func,
    const uint64_t* length,
    size_t length_size);

AOTITorchError aoti_torch_mps_dispatch_array_with_group_size(
    AOTIMetalKernelFunctionHandle func,
    const uint64_t* length,
    size_t length_size,
    const uint64_t* group_size,
    size_t group_size_size);

// Memory management functions
AOTITorchError aoti_torch_mps_malloc(void** buffer, size_t num_bytes);

AOTITorchError aoti_torch_mps_free(void* ptr);

AOTITorchError aoti_torch_mps_memcpy(
    void* buffer,
    size_t constant_offset,
    size_t bytes_read,
    size_t data_size,
    uint8_t* constants_start);

AOTITorchError aoti_torch_mps_copy_buffer(
    void* src_buffer,
    void* dst_buffer,
    size_t data_size,
    size_t src_offset,
    size_t dst_offset);

// C callback function type for command block execution
typedef void (*aoti_torch_mps_command_block_callback_t)(
    AOTIMetalKernelFunctionHandle func,
    void* user_data);

// Shared callback function for std::function trampoline
void aoti_torch_mps_shared_callback(
    AOTIMetalKernelFunctionHandle func,
    void* user_data);

// Pure C version using function pointer and user data for trampoline pattern
AOTITorchError aoti_torch_mps_run_command_block(
    AOTIMetalKernelFunctionHandle func,
    aoti_torch_mps_command_block_callback_t callback,
    void* user_data);

#ifdef __cplusplus
} // extern "C"
#endif

} // namespace metal
} // namespace backends
} // namespace executorch
