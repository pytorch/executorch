/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// AOTI kernel-dispatch C ABI for the v2 Metal backend.
//
// Symbols here are called by the AOTI .so when it has a hand-written
// shader to dispatch (shader library + kernel function + arg encoding
// + dispatch). Buffer mgmt and tensor lifecycle live in aoti_tensor.h;
// op-registry fallbacks (mm/bmm) live in aoti_ops.h.

#pragma once

#include <executorch/backends/apple/metal/runtime/shims/v2/aoti_types.h>

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

// Shader library
AOTITorchError aoti_torch_mps_create_shader_library(
    const char* metal_shader_source,
    AOTIMetalShaderLibraryHandle* library_handle);

AOTITorchError aoti_torch_mps_delete_shader_library(
    AOTIMetalShaderLibraryHandle library_handle);

AOTITorchError aoti_torch_mps_get_kernel_function(
    AOTIMetalShaderLibraryHandle library_handle,
    const char* kernel_name,
    AOTIMetalKernelFunctionHandle* function_handle);

// Kernel arg / dispatch
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

AOTITorchError aoti_torch_mps_dispatch_single(
    AOTIMetalKernelFunctionHandle func,
    uint64_t length);

AOTITorchError aoti_torch_mps_dispatch_single_with_group_size(
    AOTIMetalKernelFunctionHandle func,
    uint64_t length,
    uint64_t group_size);

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

// Command block
typedef void (*aoti_torch_mps_command_block_callback_t)(
    AOTIMetalKernelFunctionHandle func,
    void* user_data);

void aoti_torch_mps_shared_callback(
    AOTIMetalKernelFunctionHandle func,
    void* user_data);

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
