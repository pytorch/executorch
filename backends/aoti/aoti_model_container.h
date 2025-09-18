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
#include "cuda/runtime/shims/memory.h"

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
    size_t* num_constants);

using AOTInductorModelContainerGetInputNameFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** input_name);

using AOTInductorModelContainerGetNumConstantsFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants);

using AOTInductorModelContainerGetNumOutputsFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants);

using AOTInductorModelContainerRunFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle container_handle,
    AOTITensorHandle* input_handles, // array of input AOTITensorHandle; handles
                                     // are stolen; the array itself is borrowed
    size_t num_inputs,
    AOTITensorHandle*
        output_handles, // array for writing output AOTITensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    size_t n_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle);

// Global function pointers (will be loaded dynamically)
extern AOTInductorModelContainerCreateWithDeviceFunc
    AOTInductorModelContainerCreateWithDevice;
extern AOTInductorModelContainerDeleteFunc AOTInductorModelContainerDelete;
extern AOTInductorModelContainerGetNumInputsFunc
    AOTInductorModelContainerGetNumInputs;
extern AOTInductorModelContainerGetInputNameFunc
    AOTInductorModelContainerGetInputName;
extern AOTInductorModelContainerGetNumConstantsFunc
    AOTInductorModelContainerGetNumConstants;
extern AOTInductorModelContainerGetNumOutputsFunc
    AOTInductorModelContainerGetNumOutputs;
extern AOTInductorModelContainerRunFunc AOTInductorModelContainerRun;

} // extern "C"

// AOTI Delegate Handle structure
struct AOTIDelegateHandle {
  void* so_handle;
  AOTInductorModelContainerHandle container_handle;
};

} // namespace aoti
} // namespace backends
} // namespace executorch
