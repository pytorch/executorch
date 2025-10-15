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
#include <executorch/runtime/core/tensor_layout.h>
#include <unistd.h>
#include <cstdio>
#include <memory>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

// Include our shim layer headers
#include <executorch/backends/aoti/aoti_model_container.h>
#include <executorch/backends/aoti/common_shims.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/utils.h>

namespace executorch::backends::cuda {

#define LOAD_SYMBOL(name, handle)                                \
  do {                                                           \
    name = reinterpret_cast<name##Func>(dlsym(handle, #name));   \
    ET_CHECK_OR_RETURN_ERROR(                                    \
        name != nullptr, AccessFailed, "Failed to load " #name); \
  } while (0)

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

namespace {

Error parse_weight_fqns_from_processed(
    const FreeableBuffer* processed,
    std::vector<std::string>& weight_fqns) {
  if (processed == nullptr || processed->data() == nullptr ||
      processed->size() == 0) {
    return Error::Ok;
  }

  const auto* cursor = static_cast<const uint8_t*>(processed->data());
  size_t remaining = processed->size();

  auto read_uint32 = [&](uint32_t& value) -> bool {
    if (remaining < sizeof(uint32_t)) {
      return false;
    }
    std::memcpy(&value, cursor, sizeof(uint32_t));
    cursor += sizeof(uint32_t);
    remaining -= sizeof(uint32_t);
    return true;
  };

  uint32_t num_entries = 0;
  ET_CHECK_OR_RETURN_ERROR(
      read_uint32(num_entries),
      InvalidArgument,
      "Failed to read FQN count from processed bytes");

  weight_fqns.reserve(num_entries);
  for (uint32_t i = 0; i < num_entries; ++i) {
    uint32_t length = 0;
    ET_CHECK_OR_RETURN_ERROR(
        read_uint32(length),
        InvalidArgument,
        "Failed to read FQN length from processed bytes")

    ET_CHECK_OR_RETURN_ERROR(
        remaining >= length,
        InvalidArgument,
        "Processed bytes exhausted while reading FQN %u (remaining=%zu, length=%u)",
        i,
        remaining,
        length);

    const char* str_begin = reinterpret_cast<const char*>(cursor);
    weight_fqns.emplace_back(str_begin, length);
    cursor += length;
    remaining -= length;
  }

  return Error::Ok;
}

} // namespace

class ET_EXPERIMENTAL CudaBackend final
    : public ::executorch::runtime::BackendInterface {
 private:
  Error register_shared_library_functions(void* so_handle) const {
    LOAD_SYMBOL(AOTInductorModelContainerCreateWithDevice, so_handle);
    LOAD_SYMBOL(AOTInductorModelContainerDelete, so_handle);
    LOAD_SYMBOL(AOTInductorModelContainerGetNumInputs, so_handle);
    LOAD_SYMBOL(AOTInductorModelContainerGetNumOutputs, so_handle);
    LOAD_SYMBOL(AOTInductorModelContainerRun, so_handle);
    LOAD_SYMBOL(
        AOTInductorModelContainerUpdateUserManagedConstantBuffer, so_handle);

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

    std::vector<std::string> weight_fqns;
    Error parse_err = parse_weight_fqns_from_processed(processed, weight_fqns);
    if (parse_err != Error::Ok) {
      if (processed != nullptr) {
        processed->Free();
      }
      return parse_err;
    }

    std::string so_blob_key =
        method_name.empty() ? "so_blob" : method_name + "_so_blob";

    const NamedDataMap* named_data_map = context.get_named_data_map();
    auto aoti_cuda_buffer = named_data_map->get_data(so_blob_key.c_str());
    ET_CHECK_OR_RETURN_ERROR(
        aoti_cuda_buffer.ok(),
        Internal,
        "Failed to get data for key %s: 0x%x",
        so_blob_key.c_str(),
        static_cast<uint32_t>(aoti_cuda_buffer.error()));
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

    ET_CHECK_OR_RETURN_ERROR(
        outfile, AccessFailed, "Failed to write to file %s", so_path.c_str());

    // Finish writing the file to disk
    outfile.close();

    // Load the ELF using dlopen
    void* so_handle = dlopen(so_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    ET_CHECK_OR_RETURN_ERROR(
        so_handle != nullptr,
        AccessFailed,
        "Failed to load shared library: %s",
        dlerror());

    processed->Free();

    // Register all shared library functions
    ET_CHECK_OK_OR_RETURN_ERROR(register_shared_library_functions(so_handle));

    AOTInductorModelContainerHandle container_handle = nullptr;

    ET_CHECK_OK_OR_RETURN_ERROR(AOTInductorModelContainerCreateWithDevice(
        &container_handle, 1, "cuda", nullptr));

    ET_LOG(Info, "container_handle = %p", container_handle);

    AOTIDelegateHandle* handle = new AOTIDelegateHandle();
    handle->so_handle = so_handle;
    handle->so_path = so_path.string();
    handle->container_handle = container_handle;
    handle->weight_fqns = weight_fqns; // Store weight FQNs in the handle

    // Create a constant map and populate it with weights from NamedDataMap
    // Store the Tensor objects in the handle so they persist for the lifetime
    // of the container
    std::unordered_map<std::string, Tensor*> constant_map;

    for (const auto& fqn : weight_fqns) {
      // Get tensor layout (metadata) for this weight
      auto tensor_layout_result =
          named_data_map->get_tensor_layout(fqn.c_str());
      ET_CHECK_OR_RETURN_ERROR(
          tensor_layout_result.ok(),
          Internal,
          "Failed to get tensor layout for key %s: 0x%x",
          fqn.c_str(),
          static_cast<uint32_t>(tensor_layout_result.error()));

      auto weight_result = named_data_map->get_data(fqn.c_str());
      ET_CHECK_OR_RETURN_ERROR(
          weight_result.ok(),
          Internal,
          "Failed to get data for key %s: 0x%x",
          fqn.c_str(),
          static_cast<uint32_t>(weight_result.error()));

      // Store the FreeableBuffer to keep the weight data alive
      // This is critical: the FreeableBuffer owns or references the actual
      // weight data
      FreeableBuffer weight_buffer = weight_result.get();
      void* weight_data = weight_buffer.data();

      // Get tensor layout information
      const TensorLayout& layout = tensor_layout_result.get();

      // Create a Tensor from the weight data using the layout information
      // The Tensor is created as a view over the data owned by the
      // FreeableBuffer
      auto weight_tensor = std::make_unique<Tensor>(
          layout.scalar_type(),
          layout.sizes().size(),
          const_cast<Tensor::SizesType*>(layout.sizes().data()),
          weight_data,
          const_cast<Tensor::DimOrderType*>(layout.dim_order().data()),
          const_cast<Tensor::StridesType*>(layout.strides().data()));

      constant_map[fqn] = weight_tensor.get();
      handle->weight_tensors.push_back(std::move(weight_tensor));
      handle->weight_buffers.push_back(
          std::move(weight_buffer)); // Store buffer to keep data alive
    }

    // Update the container with user-managed constant buffer
    if (!constant_map.empty()) {
      AOTIRuntimeError update_err =
          AOTInductorModelContainerUpdateUserManagedConstantBuffer(
              container_handle,
              reinterpret_cast<AOTInductorConstantMapHandle>(&constant_map),
              /*use_inactive=*/false,
              /*validate_full_update=*/true);

      ET_CHECK_OR_RETURN_ERROR(
          update_err == Error::Ok,
          Internal,
          "Failed to update constant buffer with error code %d",
          update_err);

      ET_LOG(
          Info,
          "Successfully populated %zu weights into container",
          constant_map.size());
    }

    return (DelegateHandle*)handle; // Return the handle post-processing
  }

  // Once per execution
  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* handle_,
      Span<EValue*> args) const override {
    AOTIDelegateHandle* handle = (AOTIDelegateHandle*)handle_;

    // Need to re-register all the symbols from the so_handle hosted by this
    // CudaBackend instance. The reason is that these symbols are
    // static/singleton across the whole process. When we share multiple methods
    // (meaning multiple so_handle) in the same process, we need to re-register
    // the symbols from the so_handle that is being used in this execution.
    ET_CHECK_OK_OR_RETURN_ERROR(
        register_shared_library_functions(handle->so_handle));

    size_t n_inputs;
    AOTInductorModelContainerGetNumInputs(handle->container_handle, &n_inputs);

    size_t n_outputs;
    AOTInductorModelContainerGetNumOutputs(
        handle->container_handle, &n_outputs);

    ET_CHECK_OR_RETURN_ERROR(
        n_inputs + n_outputs == args.size(),
        InvalidArgument,
        "number of user input %zd and output %zd generated from AOT Inductor does not match ET runner's %zd. Exit.",
        n_inputs,
        n_outputs,
        args.size())

    // NOTE: ExecuTorch tensors are always on CPU/host memory
    // We need to create GPU copies for CUDA kernel execution
    std::vector<AOTITensorHandle> gpu_inputs(
        n_inputs); // GPU copies for kernel execution
    std::vector<AOTITensorHandle> gpu_outputs(
        n_outputs); // GPU tensors for kernel output

    // Process input tensors: ExecuTorch provides CPU tensors, create GPU
    // copies
    for (int i = 0; i < n_inputs; i++) {
      // Get tensor dimensions and properties from ExecuTorch CPU tensor
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

      ET_CHECK_OR_RETURN_ERROR(
          create_err == Error::Ok,
          Internal,
          "Failed to create GPU tensor for input %d",
          i);

      gpu_inputs[i] = gpu_input_handle;

      // Copy data from CPU to GPU
      ET_CHECK_OR_RETURN_ERROR(
          aoti_torch_copy_(gpu_inputs[i], cpu_tensor, 0) == Error::Ok,
          Internal,
          "Failed to copy input %d from CPU to GPU",
          i);
    }
    // Process output tensors: create GPU counterparts for ExecuTorch CPU
    // tensors
    for (int i = 0; i < n_outputs; i++) {
      // Get output tensor dimensions from ExecuTorch CPU tensor
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

      ET_CHECK_OR_RETURN_ERROR(
          create_err == Error::Ok,
          Internal,
          "Failed to create GPU tensor for output %d",
          i);

      gpu_outputs[i] = gpu_output_handle;
    }
    // Run AOTI container with GPU tensors
    AOTIRuntimeError error = AOTInductorModelContainerRun(
        handle->container_handle,
        gpu_inputs.data(), // Use GPU input tensors
        n_inputs,
        gpu_outputs.data(), // Use GPU output tensors
        n_outputs,
        handle->cuda_stream, // Pass the actual CUDA stream
        nullptr); // proxy_executor_handle can remain nullptr

    ET_CHECK_OR_RETURN_ERROR(
        error == Error::Ok,
        Internal,
        "AOTInductorModelContainerRun failed with error code %d",
        error);

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
    if (handle->so_handle != nullptr) {
      dlclose(handle->so_handle);
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
