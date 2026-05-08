/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// AOTI delegate handle for the v2 Metal backend.
//
// Lives in `executorch::backends::metal` (not `aoti`) to avoid a
// collision: v1's aoti_delegate_handle.h and slim's common_shims_slim.h
// both define `aoti::Tensor` to incompatible types.

#pragma once

#include <executorch/backends/apple/metal/runtime/shims/v2/aoti_types.h>
#include <cstdint>
#include <string>

namespace executorch {
namespace backends {
namespace metal {

extern "C" {

struct AOTInductorModelContainerOpaque;
using AOTInductorModelContainerHandle = AOTInductorModelContainerOpaque*;
using AOTInductorStreamHandle = void*;
using AOTIProxyExecutorHandle = void*;

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

}  // extern "C"

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

}  // namespace metal
}  // namespace backends
}  // namespace executorch
