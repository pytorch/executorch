/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <unistd.h>
#include <cstdio>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Include our shim layer headers
#include <executorch/backends/aoti/aoti_model_container.h>
#include <executorch/backends/aoti/common_shims.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/utils.h>

namespace executorch {
namespace backends {
namespace cuda {

using namespace std;
using namespace aoti;

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

class CudaBackend final : public ::executorch::runtime::BackendInterface {
 private:
  Error register_shared_library_functions(void* so_handle) const {
    AOTInductorModelContainerCreateWithDevice =
        reinterpret_cast<AOTInductorModelContainerCreateWithDeviceFunc>(
            dlsym(so_handle, "AOTInductorModelContainerCreateWithDevice"));
    if (AOTInductorModelContainerCreateWithDevice == nullptr) {
      ET_LOG(Error, "Failed to load AOTInductorModelContainerCreateWithDevice");
      return Error::AccessFailed;
    }

    AOTInductorModelContainerDelete =
        reinterpret_cast<AOTInductorModelContainerDeleteFunc>(
            dlsym(so_handle, "AOTInductorModelContainerDelete"));
    if (AOTInductorModelContainerDelete == nullptr) {
      ET_LOG(Error, "Failed to load AOTInductorModelContainerDelete");
      return Error::AccessFailed;
    }

    AOTInductorModelContainerGetNumInputs =
        reinterpret_cast<AOTInductorModelContainerGetNumInputsFunc>(
            dlsym(so_handle, "AOTInductorModelContainerGetNumInputs"));
    if (AOTInductorModelContainerGetNumInputs == nullptr) {
      ET_LOG(Error, "Failed to load AOTInductorModelContainerGetNumInputs");
      return Error::AccessFailed;
    }

    AOTInductorModelContainerGetNumOutputs =
        reinterpret_cast<AOTInductorModelContainerGetNumOutputsFunc>(
            dlsym(so_handle, "AOTInductorModelContainerGetNumOutputs"));
    if (AOTInductorModelContainerGetNumOutputs == nullptr) {
      ET_LOG(Error, "Failed to load AOTInductorModelContainerGetNumOutputs");
      return Error::AccessFailed;
    }

    AOTInductorModelContainerRun =
        reinterpret_cast<AOTInductorModelContainerRunFunc>(
            dlsym(so_handle, "AOTInductorModelContainerRun"));
    if (AOTInductorModelContainerRun == nullptr) {
      ET_LOG(Error, "Failed to load AOTInductorModelContainerRun");
      return Error::AccessFailed;
    }

    return Error::Ok;
  }

 public:
  bool is_available() const override {
    return 1;
  }

  // Once per loaded binary blob
  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed, // This will be a empty buffer
      ArrayRef<CompileSpec> compile_specs // This will be my empty list
  ) const override {
    std::string method_name;
    for (const CompileSpec& spec : compile_specs) {
      if (std::strcmp(spec.key, "method_name") == 0) {
        method_name.assign(
            static_cast<const char*>(spec.value.buffer),
            spec.value.nbytes); // no nullptr guarantee, so pass size
        break;
      }
    }

    std::string so_blob_key =
        method_name.empty() ? "so_blob" : method_name + "_so_blob";

    const NamedDataMap* named_data_map = context.get_named_data_map();
    auto aoti_cuda_buffer = named_data_map->get_data(so_blob_key.c_str());
    if (!aoti_cuda_buffer.ok()) {
      ET_LOG(
          Error,
          "Failed to get data for key %s: 0x%x",
          so_blob_key.c_str(),
          aoti_cuda_buffer.error());
      return aoti_cuda_buffer.error();
    }
    // Generate dynamic temporary file path
    filesystem::path temp_dir = filesystem::temp_directory_path();
    filesystem::path so_path =
        temp_dir / (so_blob_key + to_string(getpid()) + ".so");

    // Create a temporary file
    ofstream outfile(so_path.c_str(), ios::binary);

    // Write the ELF buffer to the temporary file
    ET_LOG(
        Info,
        "Writing %zu bytes to %s",
        aoti_cuda_buffer->size(),
        so_path.c_str());
    outfile.write(
        static_cast<const char*>(aoti_cuda_buffer->data()),
        aoti_cuda_buffer->size());

    // Finish writing the file to disk
    outfile.close();

    // Load the ELF using dlopen
    void* so_handle = dlopen(so_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (so_handle == nullptr) {
      ET_LOG(Error, "Failed to load shared library: %s", dlerror());
      return Error::AccessFailed;
    }

    processed->Free();

    // Register all shared library functions
    Error reg_err = register_shared_library_functions(so_handle);
    if (reg_err != Error::Ok) {
      return reg_err;
    }

    AOTInductorModelContainerHandle container_handle = nullptr;

    AOTIRuntimeError err = AOTInductorModelContainerCreateWithDevice(
        &container_handle, 1, "cuda", nullptr);
    if (err != Error::Ok) {
      return err;
    }
    ET_LOG(Info, "container_handle = %p", container_handle);

    AOTIDelegateHandle* handle = new AOTIDelegateHandle();
    handle->so_handle = so_handle;
    handle->container_handle = container_handle;

    // Create a CUDA stream for asynchronous execution
    cudaStream_t cuda_stream;
    ET_CUDA_CHECK_OR_RETURN_ERROR(cudaStreamCreate(&cuda_stream));
    handle->cuda_stream = static_cast<void*>(cuda_stream);
    ET_LOG(Info, "Created CUDA stream: %p", handle->cuda_stream);

    return (DelegateHandle*)handle; // Return the handle post-processing
  }

  // Once per execution
  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* handle_,
      Span<EValue*> args) const override {
    AOTIDelegateHandle* handle = (AOTIDelegateHandle*)handle_;

    size_t n_inputs;
    AOTInductorModelContainerGetNumInputs(handle->container_handle, &n_inputs);

    size_t n_outputs;
    AOTInductorModelContainerGetNumOutputs(
        handle->container_handle, &n_outputs);

    if (n_inputs + n_outputs != args.size()) {
      ET_LOG(
          Error,
          "number of user input %zd and output %zd generated from AOT Inductor does not match ET runner's %zd. Exit.",
          n_inputs,
          n_outputs,
          args.size());
      return Error::InvalidArgument;
    }

    // NOTE: ExecutorTorch tensors are always on CPU/host memory
    // We need to create GPU copies for CUDA kernel execution
    std::vector<AOTITensorHandle> gpu_inputs(
        n_inputs); // GPU copies for kernel execution
    std::vector<AOTITensorHandle> gpu_outputs(
        n_outputs); // GPU tensors for kernel output

    // Process input tensors: ExecutorTorch provides CPU tensors, create GPU
    // copies
    for (int i = 0; i < n_inputs; i++) {
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
          1, // device_type = cuda
          0, // device_index = 0
          &gpu_input_handle);

      if (create_err != Error::Ok) {
        ET_LOG(Error, "Failed to create GPU tensor for input %d", i);
        return Error::Internal;
      }

      gpu_inputs[i] = gpu_input_handle;

      // Copy data from CPU to GPU
      Error copy_err = aoti_torch_copy_(gpu_inputs[i], cpu_tensor, 0);
      if (copy_err != Error::Ok) {
        ET_LOG(Error, "Failed to copy input %d from CPU to GPU", i);
        return Error::Internal;
      }
    }
    ET_LOG(Info, "Inputs copied to GPU");
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
          1, // device_type = cuda
          0, // device_index = 0
          &gpu_output_handle);

      if (create_err != Error::Ok) {
        ET_LOG(Error, "Failed to create GPU tensor for output %d", i);
        return Error::Internal;
      }

      gpu_outputs[i] = gpu_output_handle;
    }
    ET_LOG(Info, "Outputs created on GPU");
    // Run AOTI container with GPU tensors
    AOTIRuntimeError error = AOTInductorModelContainerRun(
        handle->container_handle,
        gpu_inputs.data(), // Use GPU input tensors
        n_inputs,
        gpu_outputs.data(), // Use GPU output tensors
        n_outputs,
        handle->cuda_stream, // Pass the actual CUDA stream
        nullptr); // proxy_executor_handle can remain nullptr

    if (error != Error::Ok) {
      ET_LOG(
          Error,
          "AOTInductorModelContainerRun failed with error code %d",
          error);
      return Error::Internal;
    }

    // Copy GPU output results back to CPU output tensors
    for (int i = 0; i < n_outputs; i++) {
      auto cpu_output_tensor = &(args[i + n_inputs]->toTensor());
      // For DYNAMIC_BOUND tensors we try to resize
      ET_CHECK_OK_OR_RETURN_ERROR(
          resize_tensor(*cpu_output_tensor, gpu_outputs[i]->sizes()),
          "Error resizing tensor at output index %d",
          i);
      ET_CHECK_OK_OR_RETURN_ERROR(
          aoti_torch_copy_(cpu_output_tensor, gpu_outputs[i], 0),
          "Failed to copy GPU output %d back to CPU",
          i);
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

    return Error::Ok;
  }

  void destroy(DelegateHandle* handle_) const override {
    AOTIDelegateHandle* handle = (AOTIDelegateHandle*)handle_;

    // Destroy the CUDA stream if it exists
    if (handle->cuda_stream != nullptr) {
      cudaStream_t cuda_stream = static_cast<cudaStream_t>(handle->cuda_stream);
      cudaError_t stream_err = cudaStreamDestroy(cuda_stream);
      if (stream_err != cudaSuccess) {
        ET_LOG(
            Error,
            "Failed to destroy CUDA stream: %s",
            cudaGetErrorString(stream_err));
      } else {
        ET_LOG(Info, "Destroyed CUDA stream: %p", handle->cuda_stream);
      }
    }

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
    clear_all_tensors();
  }
};

} // namespace cuda

namespace {
auto cls = cuda::CudaBackend();
executorch::runtime::Backend backend{"CudaBackend", &cls};
static executorch::runtime::Error success_with_compiler =
    register_backend(backend);
} // namespace

} // namespace backends
} // namespace executorch
