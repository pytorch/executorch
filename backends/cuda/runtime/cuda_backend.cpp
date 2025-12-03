/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <cstdio>

#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
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

// Structure to hold cached GPU tensor data for "keep on device" optimization
struct CachedGpuData {
  void* data_ptr;           // GPU memory pointer
  size_t size_bytes;        // Total size in bytes
  int32_t scalar_type;      // Data type
  std::vector<int64_t> sizes;  // Original shape
};

// Global device cache - maps name to cached GPU data
// Using raw GPU pointers instead of tensor handles for format independence
// Note: This cache is NOT thread-safe. Callers must ensure execute() is called
// from a single thread.
static std::unordered_map<std::string, CachedGpuData> g_device_cache;

// Helper function to clear all cached GPU memory
// Should be called during backend cleanup
static void clear_device_cache() {
  for (auto& pair : g_device_cache) {
    if (pair.second.data_ptr != nullptr) {
      cudaError_t err = cudaFree(pair.second.data_ptr);
      if (err != cudaSuccess) {
        ET_LOG(
            Warning,
            "Failed to free cached GPU memory for '%s': %s",
            pair.first.c_str(),
            cudaGetErrorString(err));
      }
    }
  }
  g_device_cache.clear();
}

class ET_EXPERIMENTAL CudaBackend final
    : public ::executorch::runtime::BackendInterface {
 private:
  // Cache control options (set via set_option before execute)
  mutable int cache_output_slot_ = -1;        // Which output slot to cache (-1 = none)
  mutable std::string cache_output_name_;     // Name to cache output under
  mutable int use_cache_input_slot_ = -1;     // Which input slot to use cache for (-1 = none)
  mutable std::string use_cache_input_name_;  // Name of cached tensor to use

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

  Error set_option(
      __ET_UNUSED executorch::runtime::BackendOptionContext& context,
      const executorch::runtime::Span<executorch::runtime::BackendOption>&
          backend_options) override {
    for (size_t i = 0; i < backend_options.size(); i++) {
      const auto& option = backend_options[i];
      // Handle cache_output: "slot:name" format (e.g., "0:encoder_output")
      if (strcmp(option.key, "cache_output") == 0) {
        if (auto* arr = std::get_if<
                std::array<char, executorch::runtime::kMaxOptionValueLength>>(
                &option.value)) {
          std::string val(arr->data());
          auto colon_pos = val.find(':');
          if (colon_pos != std::string::npos) {
            try {
              cache_output_slot_ = std::stoi(val.substr(0, colon_pos));
              cache_output_name_ = val.substr(colon_pos + 1);
            } catch (const std::exception& e) {
              ET_LOG(
                  Error,
                  "Invalid cache_output format '%s': %s",
                  val.c_str(),
                  e.what());
              return Error::InvalidArgument;
            }
          }
        }
      }
      // Handle use_cache_input: "slot:name" format (e.g., "1:encoder_output")
      else if (strcmp(option.key, "use_cache_input") == 0) {
        if (auto* arr = std::get_if<
                std::array<char, executorch::runtime::kMaxOptionValueLength>>(
                &option.value)) {
          std::string val(arr->data());
          auto colon_pos = val.find(':');
          if (colon_pos != std::string::npos) {
            try {
              use_cache_input_slot_ = std::stoi(val.substr(0, colon_pos));
              use_cache_input_name_ = val.substr(colon_pos + 1);
            } catch (const std::exception& e) {
              ET_LOG(
                  Error,
                  "Invalid use_cache_input format '%s': %s",
                  val.c_str(),
                  e.what());
              return Error::InvalidArgument;
            }
          }
        }
      }
      // Handle clear_cache_input: reset input cache settings
      else if (strcmp(option.key, "clear_cache_input") == 0) {
        if (auto* val = std::get_if<bool>(&option.value)) {
          if (*val) {
            use_cache_input_slot_ = -1;
            use_cache_input_name_.clear();
          }
        }
      }
    }
    return Error::Ok;
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

    // NOTE: ExecuTorch tensors are always on CPU/host memory
    // We need to create GPU copies for CUDA kernel execution
    std::vector<AOTITensorHandle> gpu_inputs(
        n_inputs); // GPU copies for kernel execution
    std::vector<AOTITensorHandle> gpu_outputs(
        n_outputs); // GPU tensors for kernel output

    // Process input tensors: ExecuTorch provides CPU tensors, create GPU
    // copies. For cached inputs, use GPU-to-GPU copy instead of CPU-to-GPU.
    for (int i = 0; i < n_inputs; i++) {
      // Get tensor dimensions and properties from ExecuTorch CPU tensor
      auto cpu_tensor = &(args[i]->toTensor());
      auto sizes = cpu_tensor->sizes();
      auto scalar_type = cpu_tensor->scalar_type();

      // Create GPU tensor with same shape (always needed for AOTI format)
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

      // Check if this input slot should use cached GPU data
      if (i == use_cache_input_slot_ && !use_cache_input_name_.empty()) {
        auto cache_it = g_device_cache.find(use_cache_input_name_);
        if (cache_it != g_device_cache.end()) {
          const CachedGpuData& cached = cache_it->second;
          // GPU-to-GPU copy: fast DMA transfer, normalizes tensor format
          size_t numel = gpu_inputs[i]->numel();
          size_t elem_size = gpu_inputs[i]->element_size();
          size_t copy_bytes = numel * elem_size;

          ET_CHECK_OR_RETURN_ERROR(
              copy_bytes == cached.size_bytes,
              Internal,
              "Cached tensor size mismatch: expected %zu bytes, got %zu",
              copy_bytes,
              cached.size_bytes);

          cudaError_t cuda_err = cudaMemcpy(
              gpu_inputs[i]->data_ptr(),
              cached.data_ptr,
              copy_bytes,
              cudaMemcpyDeviceToDevice);

          ET_CHECK_OR_RETURN_ERROR(
              cuda_err == cudaSuccess,
              Internal,
              "Failed GPU-to-GPU copy for cached input %d: %s",
              i,
              cudaGetErrorString(cuda_err));

          // Skip the CPU-to-GPU copy below
          continue;
        }
        // Cache miss: fall through to normal CPU-to-GPU copy
      }

      // Copy data from CPU to GPU (normal path)
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
    AOTIRuntimeError error = handle->run(
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

    // Cache output GPU tensor data if requested
    // We store the raw GPU pointer for later GPU-to-GPU copy
    if (cache_output_slot_ >= 0 && cache_output_slot_ < static_cast<int>(n_outputs) &&
        !cache_output_name_.empty()) {
      auto* gpu_tensor = gpu_outputs[cache_output_slot_];
      size_t numel = gpu_tensor->numel();
      size_t elem_size = gpu_tensor->element_size();
      size_t size_bytes = numel * elem_size;

      // Allocate persistent GPU memory for the cache
      void* cache_ptr = nullptr;
      cudaError_t alloc_err = cudaMalloc(&cache_ptr, size_bytes);
      ET_CHECK_OR_RETURN_ERROR(
          alloc_err == cudaSuccess,
          Internal,
          "Failed to allocate GPU cache memory: %s",
          cudaGetErrorString(alloc_err));

      // Copy from tensor to cache (GPU-to-GPU)
      cudaError_t copy_err = cudaMemcpy(
          cache_ptr,
          gpu_tensor->data_ptr(),
          size_bytes,
          cudaMemcpyDeviceToDevice);
      if (copy_err != cudaSuccess) {
        // Free allocated memory before returning error
        cudaFree(cache_ptr);
        ET_LOG(
            Error,
            "Failed to copy output to GPU cache: %s",
            cudaGetErrorString(copy_err));
        return Error::Internal;
      }

      // Free old cache if exists
      auto old_it = g_device_cache.find(cache_output_name_);
      if (old_it != g_device_cache.end()) {
        cudaError_t free_err = cudaFree(old_it->second.data_ptr);
        if (free_err != cudaSuccess) {
          ET_LOG(
              Warning,
              "Failed to free old cached GPU memory for '%s': %s",
              cache_output_name_.c_str(),
              cudaGetErrorString(free_err));
        }
        g_device_cache.erase(old_it);
      }

      // Store in cache
      CachedGpuData cached;
      cached.data_ptr = cache_ptr;
      cached.size_bytes = size_bytes;
      cached.scalar_type = static_cast<int32_t>(gpu_tensor->scalar_type());
      auto sizes = gpu_tensor->sizes();
      cached.sizes.assign(sizes.begin(), sizes.end());
      g_device_cache[cache_output_name_] = std::move(cached);

      // Reset cache_output settings after caching
      cache_output_slot_ = -1;
      cache_output_name_.clear();
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

    // Note: use_cache_input settings are intentionally NOT reset here.
    // They persist across execute() calls to support decoder loops that
    // reuse cached encoder output. The caller should explicitly clear
    // these settings using the "clear_cache_input" option when done.

    return Error::Ok;
  }

  void destroy(DelegateHandle* handle_) const override {
    if (handle_ == nullptr) {
      return;
    }
    AOTIDelegateHandle* handle = (AOTIDelegateHandle*)handle_;

    // Clear all cached GPU memory
    clear_device_cache();

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
