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
#include "shims/utils.h"
#include "shims/shim_mps.h"

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
using executorch::runtime::NamedDataMap;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::etensor::Tensor;

class MetalBackend final : public ::executorch::runtime::BackendInterface {
 public:
  // Once in program
  MetalBackend() {
    ET_LOG(Info, "MetalBackend ctor");
  }

  bool is_available() const override {
    return 1;
  }

  // Once per loaded binary blob
  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed, // This will be a empty buffer
      ArrayRef<CompileSpec> compile_specs // This will be my empty list
  ) const override {
    ET_LOG(Info, "MetalBackend::init - Starting initialization");

    const NamedDataMap* named_data_map = context.get_named_data_map();
    ET_LOG(Info, "MetalBackend::init - Got named data map: %p", named_data_map);

    std::string so_path = "/tmp/test.so";
    std::string so_blob_key = "so_blob";
    ET_LOG(Info, "MetalBackend::init - Looking for blob key: %s", so_blob_key.c_str());

    Result<FreeableBuffer> aoti_metal_buffer =
        named_data_map->get_data(so_blob_key.c_str());
    ET_LOG(Info, "MetalBackend::init - Got buffer result");

    if (!aoti_metal_buffer.ok()) {
      ET_LOG(Error, "MetalBackend::init - Failed to get buffer: %d", (int)aoti_metal_buffer.error());
      return Error::InvalidArgument;
    }

    ET_LOG(Info, "MetalBackend::init - Buffer is OK, size: %zu", aoti_metal_buffer->size());

    if (aoti_metal_buffer->data() == nullptr) {
      ET_LOG(Error, "MetalBackend::init - Buffer data is null");
      return Error::InvalidArgument;
    }

    ET_LOG(Info, "MetalBackend::init - Buffer data pointer: %p", aoti_metal_buffer->data());

    // Create a temporary file
    ET_LOG(Info, "MetalBackend::init - Creating temp file: %s", so_path.c_str());
    std::ofstream outfile(so_path.c_str(), std::ios::binary);

    if (!outfile.is_open()) {
      ET_LOG(Error, "MetalBackend::init - Failed to create temp file");
      return Error::AccessFailed;
    }
    ET_LOG(Info, "MetalBackend::init - Temp file created successfully");

    // Write the ELF buffer to the temporary file
    size_t buffer_size = aoti_metal_buffer->size();
    ET_LOG(Info, "MetalBackend::init - About to write %zu bytes to file", buffer_size);

    // Write the buffer directly, not using sizeof(void*) multiplication
    outfile.write((char*)aoti_metal_buffer->data(), buffer_size);

    if (outfile.bad()) {
      ET_LOG(Error, "MetalBackend::init - File write failed");
      return Error::AccessFailed;
    }
    ET_LOG(Info, "MetalBackend::init - Buffer written successfully");

    // Finish writing the file to disk
    outfile.close();
    ET_LOG(Info, "MetalBackend::init - File closed successfully");

    // Load the ELF using dlopen
    void* so_handle = dlopen(so_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
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

    AOTInductorModelContainerGetInputName =
        reinterpret_cast<AOTInductorModelContainerGetInputNameFunc>(
            dlsym(so_handle, "AOTInductorModelContainerGetInputName"));
    if (AOTInductorModelContainerGetInputName == nullptr) {
      perror("dlsym AOTInductorModelContainerGetInputName");
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
    ET_LOG(Info, "MetalBackend::init - About to create AOTI container with device='mps'");

    AOTIRuntimeError err = AOTInductorModelContainerCreateWithDevice(
        &container_handle, 1, "mps", nullptr);
    ET_LOG(Info, "MetalBackend::init - AOTInductorModelContainerCreateWithDevice returned err=%d", err);

    if (err != Error::Ok) {
      ET_LOG(Error, "Failed to initialize AOTInductorModelContainer with error %d", err);
      return err;
    }
    ET_LOG(Info, "Successfully initialized container_handle = %p", container_handle);

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
    ET_LOG(Debug, "MetalBackend execute");

    AOTIDelegateHandle* handle = (AOTIDelegateHandle*)handle_;

    ET_LOG(Debug, "MetalBackend Handle generated");

    size_t n_inputs;
    AOTInductorModelContainerGetNumInputs(handle->container_handle, &n_inputs);

    size_t n_outputs;
    AOTInductorModelContainerGetNumOutputs(
        handle->container_handle, &n_outputs);

    ET_LOG(Debug, "MetalBackend n_outputs %zd generated", n_outputs);

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

    // NOTE: ExecutorTorch tensors are always on CPU/host memory
    // We need to create GPU copies for Metal kernel execution
    std::vector<AOTITensorHandle> gpu_inputs(
        n_inputs); // GPU copies for kernel execution
    std::vector<AOTITensorHandle> gpu_outputs(
        n_outputs); // GPU tensors for kernel output

    ET_LOG(Debug, "MetalBackend input/output vectors generated");

    // Process input tensors: ExecutorTorch provides CPU tensors, create GPU
    // copies
    for (int i = 0; i < n_inputs; i++) {
      ET_LOG(Debug, "Processing input %d from args to inputs vector", i);
      ET_LOG(
          Debug, "is %d input a tensor input? %d", i, int(args[i]->isTensor()));

      // Get tensor dimensions and properties from ExecutorTorch CPU tensor
      auto cpu_tensor = &(args[i]->toTensor());
      auto sizes = cpu_tensor->sizes();
      auto scalar_type = cpu_tensor->scalar_type();

      // Create GPU tensor with same shape
      std::vector<int64_t> sizes_vec(sizes.begin(), sizes.end());

      AOTITensorHandle gpu_input_handle;
      Error create_err = aoti_torch_empty_strided(
          sizes_vec.size(),
          sizes_vec.data(),
          nullptr, // use default strides
          static_cast<int32_t>(scalar_type),
          2, // device_type = mps
          0, // device_index = 0
          &gpu_input_handle);

      if (create_err != Error::Ok) {
        ET_LOG(Error, "Failed to create GPU tensor for input %d", i);
        return Error::Internal;
      }

      gpu_inputs[i] = gpu_input_handle;

      // Log the CPU tensor data before copying to GPU
      void* cpu_data = cpu_tensor->mutable_data_ptr();
      if (cpu_data && cpu_tensor->numel() > 0) {
        float* cpu_float_data = (float*)cpu_data;
        ET_LOG(Debug, "CPU input %d data before copy: [%.3f, %.3f, %.3f, ...] (numel=%zd)",
               i, cpu_float_data[0], cpu_float_data[1], cpu_float_data[2], cpu_tensor->numel());
      }

      // Copy data from CPU to GPU
      Error copy_err = aoti_torch_copy_(gpu_inputs[i], cpu_tensor, 0);
      if (copy_err != Error::Ok) {
        ET_LOG(Error, "Failed to copy input %d from CPU to GPU", i);
        return Error::Internal;
      }

      ET_LOG(Debug, "Successfully copied input %d from CPU to GPU", i);
    }

    ET_LOG(Debug, "MetalBackend GPU inputs generated");

    // Process output tensors: create GPU counterparts for ExecutorTorch CPU
    // tensors
    for (int i = 0; i < n_outputs; i++) {
      // Get output tensor dimensions from ExecutorTorch CPU tensor
      auto cpu_output_tensor = &(args[i + n_inputs]->toTensor());
      auto sizes = cpu_output_tensor->sizes();
      auto scalar_type = cpu_output_tensor->scalar_type();

      // Create GPU tensor with same shape for kernel output
      std::vector<int64_t> sizes_vec(sizes.begin(), sizes.end());

      AOTITensorHandle gpu_output_handle;
      Error create_err = aoti_torch_empty_strided(
          sizes_vec.size(),
          sizes_vec.data(),
          nullptr, // use default strides
          static_cast<int32_t>(scalar_type),
          2, // device_type = mps
          0, // device_index = 0
          &gpu_output_handle);

      if (create_err != Error::Ok) {
        ET_LOG(Error, "Failed to create GPU tensor for output %d", i);
        return Error::Internal;
      }

      gpu_outputs[i] = gpu_output_handle;
      ET_LOG(Debug, "Created GPU output tensor %d", i);
    }

    ET_LOG(Debug, "MetalBackend output generated");

    // Log tensor handles before passing to AOTI container
    ET_LOG(Debug, "Passing to AOTInductorModelContainerRun:");
    for (int i = 0; i < n_inputs; i++) {
      void* gpu_input_data = gpu_inputs[i]->mutable_data_ptr();
      ET_LOG(Debug, "  gpu_inputs[%d] = %p, data_ptr = %p",
             i, gpu_inputs[i], gpu_input_data);
    }
    for (int i = 0; i < n_outputs; i++) {
      void* gpu_output_data = gpu_outputs[i]->mutable_data_ptr();
      ET_LOG(Debug, "  gpu_outputs[%d] = %p, data_ptr = %p",
             i, gpu_outputs[i], gpu_output_data);
    }

    // Run AOTI container with GPU tensors
    AOTIRuntimeError error = AOTInductorModelContainerRun(
        handle->container_handle,
        gpu_inputs.data(), // Use GPU input tensors
        n_inputs,
        gpu_outputs.data(), // Use GPU output tensors
        n_outputs,
        nullptr, // Pass the actual Metal stream!
        nullptr); // proxy_executor_handle can remain nullptr

    if (error != Error::Ok) {
      ET_LOG(
          Error,
          "AOTInductorModelContainerRun failed with error code %d",
          error);
      return Error::Internal;
    }

    // Ensure all GPU work is completed before reading results
    Error sync_err = aoti_torch_mps_synchronize_stream();
    if (sync_err != Error::Ok) {
      ET_LOG(Error, "Failed to synchronize Metal stream after kernel execution");
      return Error::Internal;
    }

    ET_LOG(Debug, "MetalBackend running done and synchronized");

    // Copy GPU output results back to CPU output tensors
    for (int i = 0; i < n_outputs; i++) {
      auto cpu_output_tensor = &(args[i + n_inputs]->toTensor());
      Error copy_err = aoti_torch_copy_(cpu_output_tensor, gpu_outputs[i], 0);
      if (copy_err != Error::Ok) {
        ET_LOG(Error, "Failed to copy GPU output %d back to CPU", i);
        return Error::Internal;
      }
      ET_LOG(Debug, "Copied GPU output %d back to CPU", i);
    }

    // Clean up GPU tensors that we created (ExecutorTorch tensors are always
    // CPU, so all GPU tensors are our copies)
    for (int i = 0; i < n_inputs; i++) {
      // All GPU input tensors were created by us, delete them
      aoti_torch_delete_tensor_object(gpu_inputs[i]);
    }

    for (int i = 0; i < n_outputs; i++) {
      // All GPU output tensors were created by us, delete them
      aoti_torch_delete_tensor_object(gpu_outputs[i]);
    }

    ET_LOG(Debug, "MetalBackend execution completed successfully");

    return Error::Ok;
  }

  void destroy(DelegateHandle* handle_) const override {
    AOTIDelegateHandle* handle = (AOTIDelegateHandle*)handle_;

    // Delete the container BEFORE closing the shared library
    if (handle->container_handle != nullptr) {
      AOTIRuntimeError delete_result =
          AOTInductorModelContainerDelete(handle->container_handle);
      if (delete_result != Error::Ok) {
        ET_LOG(
            Error,
            "AOTInductorModelContainerDelete failed with error code %d",
            delete_result);
      }
    }

    // Now close the shared library
    if (handle->so_handle != nullptr) {
      dlclose(handle->so_handle);
    }

    free(handle);
    cleanup_memory();
    cleanup_tensor_metadata();
    cleanup_aoti_tensor_output();
    ET_LOG(Debug, "MetalBackend handle %p destroy", handle_);
  }
};

} // namespace aoti

namespace {
auto cls = aoti::MetalBackend();
executorch::runtime::Backend backend{"MetalBackend", &cls};
static executorch::runtime::Error success_with_compiler =
    register_backend(backend);
} // namespace

} // namespace backends
} // namespace executorch
