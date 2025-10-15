/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

#include <string>
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

// Forward declarations for AOT Inductor model container
struct AOTInductorModelContainerOpaque;
using AOTInductorModelContainerHandle = AOTInductorModelContainerOpaque*;
using AOTInductorStreamHandle = void*;
using AOTIProxyExecutorHandle = void*;

// Constant map handle (opaque pointer to std::unordered_map<std::string,
// AtenTensorHandle>*)
struct AOTInductorConstantMap;
using AOTInductorConstantMapHandle = AOTInductorConstantMap*;

// Function pointer types for AOT Inductor model container operations
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
    Tensor** input_handles, // array of input Tensor*; handles
                            // are stolen; the array itself is borrowed
    size_t num_inputs,
    Tensor** output_handles, // array for writing output Tensor*; handles
                             // will be stolen by the caller; the array itself
                             // is borrowed
    size_t n_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle);

using AOTInductorModelContainerUpdateUserManagedConstantBufferFunc =
    AOTIRuntimeError (*)(
        AOTInductorModelContainerHandle container_handle,
        AOTInductorConstantMapHandle constant_map_handle,
        bool use_inactive,
        bool validate_full_update);

// Global function pointers (will be loaded dynamically)
extern AOTInductorModelContainerCreateWithDeviceFunc
    AOTInductorModelContainerCreateWithDevice;
extern AOTInductorModelContainerDeleteFunc AOTInductorModelContainerDelete;
extern AOTInductorModelContainerGetNumInputsFunc
    AOTInductorModelContainerGetNumInputs;
extern AOTInductorModelContainerGetNumOutputsFunc
    AOTInductorModelContainerGetNumOutputs;
extern AOTInductorModelContainerRunFunc AOTInductorModelContainerRun;
extern AOTInductorModelContainerUpdateUserManagedConstantBufferFunc
    AOTInductorModelContainerUpdateUserManagedConstantBuffer;

// Retrieves the name of an input tensor by index from the AOTI model container.
// Needed by Metal backend
using AOTInductorModelContainerGetInputNameFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** input_name);

// Retrieves the number of constants from the AOTI model container.
// Needed by Metal backend
using AOTInductorModelContainerGetNumConstantsFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants);

// Global function pointers (will be loaded dynamically).
// Needed by Metal backend
extern AOTInductorModelContainerGetInputNameFunc
    AOTInductorModelContainerGetInputName;
extern AOTInductorModelContainerGetNumConstantsFunc
    AOTInductorModelContainerGetNumConstants;

} // extern "C"

// AOTI Delegate Handle structure
struct AOTIDelegateHandle {
  void* so_handle;
  std::string so_path;
  AOTInductorModelContainerHandle container_handle;
  void* cuda_stream; // cudaStream_t stored as void* to avoid CUDA header
                     // dependency
  std::vector<std::string> weight_fqns; // Fully qualified names of weights
  std::vector<std::unique_ptr<etensor::Tensor>>
      weight_tensors; // Storage for weight tensors
  std::vector<executorch::runtime::FreeableBuffer>
      weight_buffers; // Storage for weight data - owns the actual data
};

} // namespace aoti
} // namespace backends
} // namespace executorch
