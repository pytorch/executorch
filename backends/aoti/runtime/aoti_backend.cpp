/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

// Include our shim layer headers
#include "aoti_model_container.h"
#include "shims/memory.h"
#include "shims/tensor_attribute.h"

namespace executorch {
namespace backends {
namespace aoti {

using namespace std;

using executorch::aten::ScalarType;
using executorch::runtime::ArrayRef;
using executorch::runtime::Backend;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::etensor::Tensor;

class AOTIBackend final : public ::executorch::runtime::BackendInterface {
 public:
  // Once in program
  AOTIBackend() {
    ET_LOG(Info, "AOTIBackend ctor");
  }

  bool is_available() const override {
    return 1;
  }

  // Once per loaded binary blob
  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed, // This will be the buffer from aoti_backend
      ArrayRef<CompileSpec> compile_specs // This will be my empty list
  ) const override {
    const char* so_path = static_cast<const char*>(processed->data());

    printf("so path: %s\n", so_path);

    // Load the ELF using dlopen
    void* so_handle = dlopen(so_path, RTLD_LAZY | RTLD_LOCAL);
    if (so_handle == nullptr) {
      std::cout << dlerror() << std::endl;
      return Error::AccessFailed;
    }

    processed->Free();

    AOTInductorModelContainerCreateWithDevice =
        reinterpret_cast<AOTInductorModelContainerCreateWithDeviceFunc>(
            dlsym(so_handle, "AOTInductorModelContainerCreateWithDevice"));
    if (AOTInductorModelContainerCreateWithDevice == nullptr) {
      perror("dlsym1");
      return Error::AccessFailed;
    }
    AOTInductorModelContainerDelete =
        reinterpret_cast<AOTInductorModelContainerDeleteFunc>(
            dlsym(so_handle, "AOTInductorModelContainerDelete"));
    if (AOTInductorModelContainerDelete == nullptr) {
      perror("dlsym2");
      return Error::AccessFailed;
    }
    AOTInductorModelContainerGetNumInputs =
        reinterpret_cast<AOTInductorModelContainerGetNumInputsFunc>(
            dlsym(so_handle, "AOTInductorModelContainerGetNumInputs"));
    if (AOTInductorModelContainerGetNumInputs == nullptr) {
      perror("dlsym3");
      return Error::AccessFailed;
    }

    AOTInductorModelContainerGetNumConstants =
        reinterpret_cast<AOTInductorModelContainerGetNumConstantsFunc>(
            dlsym(so_handle, "AOTInductorModelContainerGetNumConstants"));
    if (AOTInductorModelContainerGetNumConstants == nullptr) {
      perror("dlsym AOTInductorModelContainerGetNumConstants");
      return Error::AccessFailed;
    }

    AOTInductorModelContainerGetNumOutputs =
        reinterpret_cast<AOTInductorModelContainerGetNumOutputsFunc>(
            dlsym(so_handle, "AOTInductorModelContainerGetNumOutputs"));
    if (AOTInductorModelContainerGetNumOutputs == nullptr) {
      perror("dlsym4");
      return Error::AccessFailed;
    }
    AOTInductorModelContainerRun =
        reinterpret_cast<AOTInductorModelContainerRunFunc>(
            dlsym(so_handle, "AOTInductorModelContainerRun"));
    if (AOTInductorModelContainerRun == nullptr) {
      perror("dlsym5");
      return Error::AccessFailed;
    }

    AOTInductorModelContainerHandle container_handle = nullptr;

    AOTIRuntimeError err = AOTInductorModelContainerCreateWithDevice(
        &container_handle, 1, "cuda", nullptr);
    if (err != Error::Ok) {
      return err;
    }
    printf("container_handle = %p\n", container_handle);

    AOTIDelegateHandle* handle = new AOTIDelegateHandle();
    handle->so_handle = so_handle;
    handle->container_handle = container_handle;
    return (DelegateHandle*)handle; // Return the handle post-processing
  }

  // Once per execution
  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* handle_,
      Span<EValue*> args) const override {
    ET_LOG(Debug, "AOTIBackend execute");

    AOTIDelegateHandle* handle = (AOTIDelegateHandle*)handle_;

    ET_LOG(Debug, "AOTIBackend Handle generated");

    size_t n_inputs, n_constants;
    AOTInductorModelContainerGetNumInputs(handle->container_handle, &n_inputs);

    AOTInductorModelContainerGetNumConstants(
        handle->container_handle, &n_constants);
    size_t n_user_inputs = n_inputs - n_constants;

    if (n_user_inputs != n_inputs) {
      ET_LOG(
          Error,
          "number of user input does not match number of inputs. n_user_inputs %zd, n_constant %zd, n_inputs %zd. Exit.",
          n_user_inputs,
          n_constants,
          n_inputs);
      return Error::InvalidArgument;
    }

    ET_LOG(
        Debug,
        "AOTIBackend n_inputs %zd generated, where %zd is constant input, %zd is user input",
        n_inputs,
        n_constants,
        n_user_inputs);

    size_t n_outputs;
    AOTInductorModelContainerGetNumOutputs(
        handle->container_handle, &n_outputs);

    ET_LOG(Debug, "AOTIBackend n_outputs %zd generated", n_outputs);

    if (n_inputs + n_outputs != args.size()) {
      ET_LOG(
          Error,
          "number of user input %zd and output %zd generated from AOT Inductor does not match ET runner's %zd. Exit.",
          n_inputs,
          n_outputs,
          args.size());
      return Error::InvalidArgument;
    }

    ET_LOG(
        Debug,
        "number of user input %zd and output %zd generated from AOT Inductor matches ET runner's %zd.",
        n_inputs,
        n_outputs,
        args.size());

    std::vector<AOTITensorHandle> inputs(n_inputs);
    std::vector<AOTITensorHandle> outputs(n_outputs);

    ET_LOG(Debug, "AOTIBackend input/output vectors generated");

    for (int i = 0; i < n_inputs; i++) {
      ET_LOG(Debug, "Copying input %d from args to inputs vector", i);
      ET_LOG(
          Debug, "is %d input a tensor input? %d", i, int(args[i]->isTensor()));
      inputs[i] = &(args[i]->toTensor());
    }

    ET_LOG(Debug, "AOTIBackend input generated");

    for (int i = 0; i < n_outputs; i++) {
      outputs[i] = &(args[i + n_inputs]->toTensor());
    }

    ET_LOG(Debug, "AOTIBackend output generated");

    // Create a CUDA stream for this execution
    cudaStream_t cuda_stream;
    cudaError_t stream_err = cudaStreamCreate(&cuda_stream);
    if (stream_err != cudaSuccess) {
      ET_LOG(
          Error,
          "Failed to create CUDA stream: %s",
          cudaGetErrorString(stream_err));
      return Error::Internal;
    }

    ET_LOG(Debug, "Created CUDA stream: %p", cuda_stream);

    // Run AOTI container with the stream (AOTI will create its own stream guard
    // internally)
    AOTIRuntimeError error = AOTInductorModelContainerRun(
        handle->container_handle,
        inputs.data(),
        n_inputs,
        outputs.data(),
        n_outputs,
        cuda_stream, // Pass the actual CUDA stream!
        nullptr); // proxy_executor_handle can remain nullptr

    if (error != Error::Ok) {
      ET_LOG(
          Error,
          "AOTInductorModelContainerRun failed with error code %d",
          error);
      return Error::Internal;
    }

    ET_LOG(Debug, "AOTIBackend running done");

    // Synchronize and destroy the CUDA stream
    cudaError_t sync_err = cudaStreamSynchronize(cuda_stream);
    if (sync_err != cudaSuccess) {
      ET_LOG(
          Error,
          "Failed to synchronize CUDA stream: %s",
          cudaGetErrorString(sync_err));
      // Continue anyway to avoid fatal errors
    }

    cudaStreamDestroy(cuda_stream);
    ET_LOG(Debug, "CUDA stream synchronized and destroyed");

    // Still need to copy the output to args, because they are malloc'ed but
    // not using the data_ptr from outputs.
    for (int i = 0; i < n_outputs; i++) {
      auto args_out = args[i + n_inputs]->toTensor();
      aoti_torch_copy_(&args_out, outputs[i], 0);
    }

    ET_LOG(Debug, "AOTIBackend output copied");

    return Error::Ok;
  }

  void destroy(DelegateHandle* handle_) const override {
    ET_LOG(Debug, "AOTIBackend handle %p destroy", handle_);
    AOTIDelegateHandle* handle = (AOTIDelegateHandle*)handle_;
    dlclose(handle->so_handle);
    AOTInductorModelContainerDelete(handle->container_handle);
    free(handle);
    cleanup_memory();
    cleanup_tensor_metadata();
  }
};

} // namespace aoti

namespace {
auto cls = aoti::AOTIBackend();
executorch::runtime::Backend backend{"AotiBackend", &cls};
static executorch::runtime::Error success_with_compiler =
    register_backend(backend);
} // namespace

} // namespace backends
} // namespace executorch
