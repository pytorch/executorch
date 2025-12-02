/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <cstdio>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// Include our shim layer headers
#include <executorch/backends/aoti/aoti_delegate_handle.h>
#include <executorch/backends/aoti/common_shims.h>
#include <executorch/backends/cuda/runtime/platform/platform.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/utils.h>

namespace executorch::backends::cuda {

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

class ET_EXPERIMENTAL CudaBackend final
    : public ::executorch::runtime::BackendInterface {
 private:
  Error load_function_pointers_into_handle(
      void* so_handle,
      AOTIDelegateHandle* handle) const {
#define LOAD_SYMBOL(member, name)                                    \
  do {                                                               \
    auto symbol_res = get_function(so_handle, #name);                \
    if (!symbol_res.ok()) {                                          \
      return symbol_res.error();                                     \
    }                                                                \
    handle->member = reinterpret_cast<name##Func>(symbol_res.get()); \
  } while (0)

    LOAD_SYMBOL(create_with_device, AOTInductorModelContainerCreateWithDevice);

    LOAD_SYMBOL(delete_container, AOTInductorModelContainerDelete);

    LOAD_SYMBOL(get_num_inputs, AOTInductorModelContainerGetNumInputs);

    LOAD_SYMBOL(get_num_outputs, AOTInductorModelContainerGetNumOutputs);

    LOAD_SYMBOL(run, AOTInductorModelContainerRun);
#undef LOAD_SYMBOL

    auto symbol_res =
        get_function(so_handle, "AOTInductorModelUpdateConstantsFromBlob");
    if (symbol_res.ok()) {
      handle->update_constants_from_blob =
          reinterpret_cast<AOTInductorModelUpdateConstantsFromBlobFunc>(
              symbol_res.get());
    } else {
      ET_LOG(
          Info,
          "Failed to load AOTInductorModelUpdateConstantsFromBlob. This .so is probably compiled on an old version of torch (<2.9.0)");
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
    auto aoti_dso_buffer = named_data_map->get_data(so_blob_key.c_str());
    ET_CHECK_OR_RETURN_ERROR(
        aoti_dso_buffer.ok(),
        Internal,
        "Failed to get data for key %s: 0x%x",
        so_blob_key.c_str(),
        static_cast<uint32_t>(aoti_dso_buffer.error()));

    // Generate dynamic temporary file path
    filesystem::path temp_dir = filesystem::temp_directory_path();
    filesystem::path so_path =
        temp_dir / (so_blob_key + to_string(get_process_id()) + ".so");

    // Create a temporary file
    ofstream outfile(so_path, ios::binary);

    // Write the ELF buffer to the temporary file
    ET_LOG(
        Info,
        "Writing %zu bytes to %s",
        aoti_dso_buffer->size(),
        so_path.c_str());

    outfile.write(
        static_cast<const char*>(aoti_dso_buffer->data()),
        aoti_dso_buffer->size());

    ET_CHECK_OR_RETURN_ERROR(
        outfile, AccessFailed, "Failed to write to file %s", so_path.c_str());

    // Finish writing the file to disk
    outfile.close();

    // Free the buffer immediately after writing to disk
    aoti_dso_buffer->Free();
    // Load the lib
    Result<void*> lib_handle_res = load_library(so_path);
    if (!lib_handle_res.ok()) {
      return lib_handle_res.error();
    }
    void* lib_handle = lib_handle_res.get();

    processed->Free();

    // Create handle and load function pointers into it
    AOTIDelegateHandle* handle = new AOTIDelegateHandle();
    handle->so_handle = lib_handle;
    handle->so_path = so_path.string();

    // Load function pointers specific to this handle's shared library
    ET_CHECK_OK_OR_RETURN_ERROR(
        load_function_pointers_into_handle(lib_handle, handle));

    AOTInductorModelContainerHandle container_handle = nullptr;

    ET_CHECK_OK_OR_RETURN_ERROR(
        handle->create_with_device(&container_handle, 1, "cuda", nullptr));

    ET_LOG(Info, "container_handle = %p", container_handle);

    handle->container_handle = container_handle;

    // Look into named data map for constant data
    std::string weights_blob_key =
        method_name.empty() ? "weights_blob" : method_name + "_weights_blob";
    auto buffer_res = named_data_map->get_data(weights_blob_key.c_str());
    if (buffer_res.ok() && handle->update_constants_from_blob != nullptr) {
      ET_LOG(Info, "Found %s in named data map", weights_blob_key.c_str());
      const void* weights_blob = buffer_res->data();
      // Feed the weights blob into the container. Under the hood it's copying
      // weights, so we should free the buffer immediately.
      ET_CHECK_OK_OR_RETURN_ERROR(handle->update_constants_from_blob(
          handle->container_handle, static_cast<const uint8_t*>(weights_blob)));
      buffer_res->Free();
    }
    // Create a CUDA stream for asynchronous execution
    cudaStream_t cuda_stream;
    ET_CUDA_CHECK_OR_RETURN_ERROR(cudaStreamCreate(&cuda_stream));
    handle->cuda_stream = static_cast<void*>(cuda_stream);

    return (DelegateHandle*)handle; // Return the handle post-processing
  }

  // Once per execution
  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* handle_,
      Span<EValue*> args) const override {
    AOTIDelegateHandle* handle = (AOTIDelegateHandle*)handle_;
    cudaStream_t stream = static_cast<cudaStream_t>(handle->cuda_stream);

    size_t n_inputs;
    handle->get_num_inputs(handle->container_handle, &n_inputs);

    size_t n_outputs;
    handle->get_num_outputs(handle->container_handle, &n_outputs);

    ET_CHECK_OR_RETURN_ERROR(
        n_inputs + n_outputs == args.size(),
        InvalidArgument,
        "number of user input %zd and output %zd generated from AOT Inductor does not match ET runner's %zd. Exit.",
        n_inputs,
        n_outputs,
        args.size())

    // NOTE: ExecuTorch tensors are always on CPU/host memory (pageable)
    // We use pinned staging buffers for efficient async transfers:
    // CPU (pageable) -> Pinned -> GPU -> Pinned -> CPU (pageable)
    std::vector<AOTITensorHandle> pinned_inputs(n_inputs);
    std::vector<AOTITensorHandle> gpu_inputs(n_inputs);
    std::vector<AOTITensorHandle> gpu_outputs(n_outputs);
    std::vector<AOTITensorHandle> pinned_outputs(n_outputs);

    // Process input tensors: create pinned staging buffers and GPU tensors
    for (int i = 0; i < n_inputs; i++) {
      auto cpu_tensor = &(args[i]->toTensor());
      auto sizes = cpu_tensor->sizes();
      auto scalar_type = cpu_tensor->scalar_type();
      std::vector<int64_t> sizes_vec(sizes.begin(), sizes.end());

      // Create pinned staging buffer
      AOTITensorHandle pinned_input_handle;
      ET_CHECK_OR_RETURN_ERROR(
          aoti_torch_empty_strided_pinned(
              sizes_vec.size(),
              sizes_vec.data(),
              nullptr, // use default strides
              static_cast<int32_t>(scalar_type),
              &pinned_input_handle) == Error::Ok,
          Internal,
          "Failed to create pinned staging buffer for input %d",
          i);
      pinned_inputs[i] = pinned_input_handle;

      // Create GPU tensor
      AOTITensorHandle gpu_input_handle;
      ET_CHECK_OR_RETURN_ERROR(
          aoti_torch_empty_strided(
              sizes_vec.size(),
              sizes_vec.data(),
              nullptr, // use default strides
              static_cast<int32_t>(scalar_type),
              1, // device_type = cuda
              0, // device_index = 0
              &gpu_input_handle) == Error::Ok,
          Internal,
          "Failed to create GPU tensor for input %d",
          i);
      gpu_inputs[i] = gpu_input_handle;

      // Copy from ExecuTorch CPU to pinned buffer (fast memcpy)
      std::memcpy(
          pinned_inputs[i]->mutable_data_ptr(),
          cpu_tensor->data_ptr(),
          cpu_tensor->nbytes());

      // Async copy from pinned to GPU (truly async with DMA)
      ET_CHECK_OR_RETURN_ERROR(
          aoti_torch_copy_async(gpu_inputs[i], pinned_inputs[i], stream) ==
              Error::Ok,
          Internal,
          "Failed to async copy input %d from pinned to GPU",
          i);
    }

    // Process output tensors: create GPU tensors and pinned staging buffers
    for (int i = 0; i < n_outputs; i++) {
      auto cpu_output_tensor = &(args[i + n_inputs]->toTensor());
      auto sizes = cpu_output_tensor->sizes();
      auto scalar_type = cpu_output_tensor->scalar_type();
      std::vector<int64_t> sizes_vec(sizes.begin(), sizes.end());

      // Create GPU tensor for kernel output
      AOTITensorHandle gpu_output_handle;
      ET_CHECK_OR_RETURN_ERROR(
          aoti_torch_empty_strided(
              sizes_vec.size(),
              sizes_vec.data(),
              nullptr, // use default strides
              static_cast<int32_t>(scalar_type),
              1, // device_type = cuda
              0, // device_index = 0
              &gpu_output_handle) == Error::Ok,
          Internal,
          "Failed to create GPU tensor for output %d",
          i);
      gpu_outputs[i] = gpu_output_handle;

      // Create pinned staging buffer for output
      AOTITensorHandle pinned_output_handle;
      ET_CHECK_OR_RETURN_ERROR(
          aoti_torch_empty_strided_pinned(
              sizes_vec.size(),
              sizes_vec.data(),
              nullptr, // use default strides
              static_cast<int32_t>(scalar_type),
              &pinned_output_handle) == Error::Ok,
          Internal,
          "Failed to create pinned staging buffer for output %d",
          i);
      pinned_outputs[i] = pinned_output_handle;
    }

    // Run AOTI container with GPU tensors
    // Note: kernel is queued on the same stream as H2D copies,
    // so it will automatically wait for copies to complete
    AOTIRuntimeError error = handle->run(
        handle->container_handle,
        gpu_inputs.data(),
        n_inputs,
        gpu_outputs.data(),
        n_outputs,
        handle->cuda_stream,
        nullptr);

    ET_CHECK_OR_RETURN_ERROR(
        error == Error::Ok,
        Internal,
        "AOTInductorModelContainerRun failed with error code %d",
        error);

    // Async copy GPU outputs to pinned staging buffers (truly async with DMA)
    for (int i = 0; i < n_outputs; i++) {
      ET_CHECK_OR_RETURN_ERROR(
          aoti_torch_copy_async(pinned_outputs[i], gpu_outputs[i], stream) ==
              Error::Ok,
          Internal,
          "Failed to async copy GPU output %d to pinned buffer",
          i);
    }

    // Synchronize stream to ensure all async operations complete
    ET_CUDA_CHECK_OR_RETURN_ERROR(cudaStreamSynchronize(stream));

    // Copy from pinned buffers to ExecuTorch CPU output tensors (fast memcpy)
    for (int i = 0; i < n_outputs; i++) {
      auto cpu_output_tensor = &(args[i + n_inputs]->toTensor());
      // For DYNAMIC_BOUND tensors we try to resize
      ET_CHECK_OK_OR_RETURN_ERROR(
          resize_tensor(*cpu_output_tensor, gpu_outputs[i]->sizes()),
          "Error resizing tensor at output index %d",
          i);
      std::memcpy(
          cpu_output_tensor->mutable_data_ptr(),
          pinned_outputs[i]->data_ptr(),
          pinned_outputs[i]->nbytes());
    }

    return Error::Ok;
  }

  void destroy(DelegateHandle* handle_) const override {
    if (handle_ == nullptr) {
      return;
    }
    AOTIDelegateHandle* handle = (AOTIDelegateHandle*)handle_;

    // Destroy the CUDA stream if it exists
    if (handle->cuda_stream != nullptr) {
      cudaStream_t cuda_stream = static_cast<cudaStream_t>(handle->cuda_stream);
      cudaError_t stream_err = cudaStreamDestroy(cuda_stream);
      ET_CHECK_OR_LOG_ERROR(
          stream_err == cudaSuccess,
          "Failed to destroy CUDA stream: %s",
          cudaGetErrorString(stream_err));
      handle->cuda_stream = nullptr;
    }

    // NOTE: AOTInductorModelContainerDelete does not work correctly with
    // multiple .so files. Deleting one container frees shared resources,
    // which causes segmentation faults when attempting to delete other
    // containers. As a workaround, we skip explicit container deletion
    // and defer cleanup to the OS.
    // TODO(gasoonjia): Find a proper solution for safe container deletion.
    // AOTInductorModelContainerDelete(handle->container_handle);

    // Now close the shared library
    auto err = Error::Ok;
    if (handle->so_handle != nullptr) {
      err = close_library(handle->so_handle);
    }

    // Remove the temporary shared library file
    if (!handle->so_path.empty()) {
      std::error_code remove_error;
      std::filesystem::remove(handle->so_path, remove_error);
      ET_CHECK_OR_LOG_ERROR(
          !remove_error,
          "Failed to remove temporary shared library %s: %s",
          handle->so_path.c_str(),
          remove_error.message().c_str());
    }

    delete handle;
    clear_all_tensors();
  }
};

} // namespace executorch::backends::cuda

namespace executorch::backends {
namespace {
auto cls = cuda::CudaBackend();
executorch::runtime::Backend backend{"CudaBackend", &cls};
static executorch::runtime::Error success_with_compiler =
    register_backend(backend);
} // namespace
} // namespace executorch::backends
