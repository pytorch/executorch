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

// Retrieves the name of an input tensor by index from the AOTI model container.
using AOTInductorModelContainerGetInputNameFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** input_name);

// Retrieves the number of constants from the AOTI model container.
using AOTInductorModelContainerGetNumConstantsFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants);

// Update the model container with the constant tensors
using AOTInductorModelUpdateConstantsFromBlobFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle container_handle,
    const uint8_t* weight_blob_ptr);

} // extern "C"

// AOTI Delegate Handle structure
struct AOTIDelegateHandle {
  void* so_handle;
  std::string so_path;
  AOTInductorModelContainerHandle container_handle;
  void* cuda_stream; // cudaStream_t stored as void* to avoid CUDA header
                     // dependency
  bool owns_stream{true};

  // Function pointers specific to this handle's shared library
  AOTInductorModelContainerCreateWithDeviceFunc create_with_device;
  AOTInductorModelContainerDeleteFunc delete_container;
  AOTInductorModelContainerGetNumInputsFunc get_num_inputs;
  AOTInductorModelContainerGetNumOutputsFunc get_num_outputs;
  AOTInductorModelContainerRunFunc run;
  AOTInductorModelUpdateConstantsFromBlobFunc update_constants_from_blob;
};

} // namespace aoti
} // namespace backends
} // namespace executorch
