/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SlimTensor-flavored AOTI delegate handle for the v2 Metal backend.
//
// Mirrors aoti/aoti_delegate_handle.h, but lives in
// `executorch::backends::metal` so it doesn't collide with the v1 header's
// `aoti::Tensor = etensor::Tensor` typedef. (v1's header and slim's
// `common_shims_slim.h` both define `aoti::Tensor` to different types,
// so they can't be in the same TU.)
//
// At the ABI level the AOTI .so calls these function pointers with
// opaque pointers — the underlying type doesn't matter. We use SlimTensor
// here so the rest of the v2 code stays consistent.

#pragma once

#include <executorch/backends/apple/metal/runtime/shims/v2/aoti_types.h>
#include <cstdint>
#include <string>

namespace executorch {
namespace backends {
namespace metal {

extern "C" {

// Forward declarations for AOT Inductor model container
struct AOTInductorModelContainerOpaque;
using AOTInductorModelContainerHandle = AOTInductorModelContainerOpaque*;
using AOTInductorStreamHandle = void*;
using AOTIProxyExecutorHandle = void*;

// Function pointer types for AOT Inductor model container operations.
// Use Tensor (= SlimTensor) for the run() handles.
using AOTInductorModelContainerCreateWithDeviceFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
    const char* cubin_dir);

using AOTInductorModelContainerDeleteFunc =
    AOTIRuntimeError (*)(AOTInductorModelContainerHandle container_handle);

using AOTInductorModelContainerGetNumInputsFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_inputs);

using AOTInductorModelContainerGetNumOutputsFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_outputs);

using AOTInductorModelContainerRunFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle container_handle,
    Tensor** input_handles,
    size_t num_inputs,
    Tensor** output_handles,
    size_t n_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle);

using AOTInductorModelUpdateConstantsFromBlobFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle container_handle,
    const uint8_t* weight_blob_ptr);

} // extern "C"

// AOTI Delegate Handle structure (v2: minimal subset used by
// metal_backend_v2.cpp; constant-management fields omitted).
struct AOTIDelegateHandle {
  void* so_handle;
  std::string so_path;
  AOTInductorModelContainerHandle container_handle;
  std::string method_name;

  AOTInductorModelContainerCreateWithDeviceFunc create_with_device;
  AOTInductorModelContainerDeleteFunc delete_container;
  AOTInductorModelContainerGetNumInputsFunc get_num_inputs;
  AOTInductorModelContainerGetNumOutputsFunc get_num_outputs;
  AOTInductorModelContainerRunFunc run;
  AOTInductorModelUpdateConstantsFromBlobFunc update_constants_from_blob;
};

} // namespace metal
} // namespace backends
} // namespace executorch
