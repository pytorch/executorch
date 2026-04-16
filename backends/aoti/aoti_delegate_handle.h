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

// Opaque types for AOTI constant management.
// AtenTensorOpaque wraps at::Tensor* in the AOTI runtime — distinct from
// AOTITensorHandle which wraps executorch::runtime::etensor::Tensor*.
struct AtenTensorOpaque;
using AtenTensorHandle = AtenTensorOpaque*;

struct AOTInductorConstantMap;
using AOTInductorConstantMapHandle = AOTInductorConstantMap*;

struct AOTInductorConstantMapEntry {
  const char* name;
  AtenTensorHandle handle;
};

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

// Retrieves a constant's AOTI internal name by index.
using AOTInductorModelContainerGetConstantNameFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** name);

// Retrieves a constant's original fully-qualified name by index.
using AOTInductorModelContainerGetConstantOriginalFQNFunc =
    AOTIRuntimeError (*)(
        AOTInductorModelContainerHandle container_handle,
        size_t idx,
        const char** original_fqn);

// Retrieves a constant's data size in bytes by index.
using AOTInductorModelContainerGetConstantDataSizeFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    size_t* data_size);

// Retrieves whether a constant was produced by constant folding.
using AOTInductorModelContainerGetConstantFromFoldedFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    bool* from_folded);

// Retrieves the total size of the constants blob.
using AOTInductorModelContainerGetConstantsBlobSizeFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle container_handle,
    uint64_t* ret_size);

// Extracts the constants map from the container (active or inactive buffer).
// constant_map_handle should point to a
// std::unordered_map<std::string, AtenTensorHandle>.
using AOTInductorModelContainerExtractConstantsMapFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive);

// Updates the container's constants with user-managed tensor handles.
// DLL-boundary safe — uses a flat C array instead of std::unordered_map.
using AOTInductorModelContainerUpdateUserManagedConstantBufferPairsFunc =
    AOTIRuntimeError (*)(
        AOTInductorModelContainerHandle container_handle,
        const AOTInductorConstantMapEntry* pairs,
        size_t num_pairs,
        bool use_inactive,
        bool validate_full_update);

} // extern "C"

// AOTI Delegate Handle structure
struct AOTIDelegateHandle {
  void* so_handle;
  std::string so_path;
  AOTInductorModelContainerHandle container_handle;
  std::string method_name;

  // Function pointers specific to this handle's shared library
  AOTInductorModelContainerCreateWithDeviceFunc create_with_device;
  AOTInductorModelContainerDeleteFunc delete_container;
  AOTInductorModelContainerGetNumInputsFunc get_num_inputs;
  AOTInductorModelContainerGetNumOutputsFunc get_num_outputs;
  AOTInductorModelContainerRunFunc run;
  AOTInductorModelUpdateConstantsFromBlobFunc update_constants_from_blob;

  // Constant management function pointers (for cross-method buffer sharing)
  AOTInductorModelContainerGetNumConstantsFunc get_num_constants;
  AOTInductorModelContainerGetConstantNameFunc get_constant_name;
  AOTInductorModelContainerGetConstantOriginalFQNFunc get_constant_original_fqn;
  AOTInductorModelContainerGetConstantDataSizeFunc get_constant_data_size;
  AOTInductorModelContainerGetConstantFromFoldedFunc get_constant_from_folded;
  AOTInductorModelContainerGetConstantsBlobSizeFunc get_constants_blob_size;
  AOTInductorModelContainerExtractConstantsMapFunc extract_constants_map;
  AOTInductorModelContainerUpdateUserManagedConstantBufferPairsFunc
      update_user_managed_constant_buffer_pairs;
};

} // namespace aoti
} // namespace backends
} // namespace executorch
