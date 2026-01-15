/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dlfcn.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// Include AOTI common headers (from aoti_common library)
#include <executorch/backends/aoti/aoti_delegate_handle.h>
#include <executorch/backends/aoti/common_shims.h>

// Include our Metal-specific shim layer headers
#include <executorch/backends/apple/metal/runtime/shims/et_metal.h>
#include <executorch/backends/apple/metal/runtime/shims/memory.h>
#include <executorch/backends/apple/metal/runtime/shims/shim_mps.h>
#include <executorch/backends/apple/metal/runtime/shims/tensor_attribute.h>
#include <executorch/backends/apple/metal/runtime/shims/utils.h>
#include <executorch/backends/apple/metal/runtime/stats.h>

namespace executorch::backends::metal {

// Timing statistics for execute() calls
static double g_execute_total_ms = 0.0;
static int64_t g_execute_call_count = 0;

// Timing statistics for init() calls
static double g_init_total_ms = 0.0;
static int64_t g_init_call_count = 0;

// Per-method timing statistics (for both init and execute)
struct MethodStats {
  double total_ms = 0.0;
  int64_t call_count = 0;
};
static std::unordered_map<std::string, MethodStats> g_method_stats;
static std::unordered_map<std::string, MethodStats> g_init_method_stats;

// Accessor functions for execute timing statistics
double get_metal_backend_execute_total_ms() {
  return g_execute_total_ms;
}

int64_t get_metal_backend_execute_call_count() {
  return g_execute_call_count;
}

// Accessor functions for init timing statistics
double get_metal_backend_init_total_ms() {
  return g_init_total_ms;
}

int64_t get_metal_backend_init_call_count() {
  return g_init_call_count;
}

void reset_metal_backend_execute_stats() {
  g_execute_total_ms = 0.0;
  g_execute_call_count = 0;
  g_init_total_ms = 0.0;
  g_init_call_count = 0;
  g_method_stats.clear();
  g_init_method_stats.clear();
}

std::unordered_map<std::string, std::pair<double, int64_t>>
get_metal_backend_per_method_stats() {
  std::unordered_map<std::string, std::pair<double, int64_t>> result;
  for (const auto& entry : g_method_stats) {
    result[entry.first] = {entry.second.total_ms, entry.second.call_count};
  }
  return result;
}

std::unordered_map<std::string, std::pair<double, int64_t>>
get_metal_backend_init_per_method_stats() {
  std::unordered_map<std::string, std::pair<double, int64_t>> result;
  for (const auto& entry : g_init_method_stats) {
    result[entry.first] = {entry.second.total_ms, entry.second.call_count};
  }
  return result;
}

#define LOAD_SYMBOL(handle, member, name, so_handle)                        \
  do {                                                                      \
    handle->member = reinterpret_cast<name##Func>(dlsym(so_handle, #name)); \
    ET_CHECK_OR_RETURN_ERROR(                                               \
        handle->member != nullptr, AccessFailed, "Failed to load " #name);  \
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

class ET_EXPERIMENTAL MetalBackend final
    : public ::executorch::runtime::BackendInterface {
 private:
  Error load_function_pointers_into_handle(
      void* so_handle,
      AOTIDelegateHandle* handle) const {
    ET_LOG(
        Debug,
        "MetalBackend::load_function_pointers_into_handle - Loading symbols");

    LOAD_SYMBOL(
        handle,
        create_with_device,
        AOTInductorModelContainerCreateWithDevice,
        so_handle);
    ET_LOG(
        Debug,
        "MetalBackend::load_function_pointers_into_handle - Loaded AOTInductorModelContainerCreateWithDevice");

    LOAD_SYMBOL(
        handle, delete_container, AOTInductorModelContainerDelete, so_handle);
    ET_LOG(
        Debug,
        "MetalBackend::load_function_pointers_into_handle - Loaded AOTInductorModelContainerDelete");

    LOAD_SYMBOL(
        handle,
        get_num_inputs,
        AOTInductorModelContainerGetNumInputs,
        so_handle);
    ET_LOG(
        Debug,
        "MetalBackend::load_function_pointers_into_handle - Loaded AOTInductorModelContainerGetNumInputs");

    LOAD_SYMBOL(
        handle,
        get_num_outputs,
        AOTInductorModelContainerGetNumOutputs,
        so_handle);
    ET_LOG(
        Debug,
        "MetalBackend::load_function_pointers_into_handle - Loaded AOTInductorModelContainerGetNumOutputs");

    LOAD_SYMBOL(handle, run, AOTInductorModelContainerRun, so_handle);
    ET_LOG(
        Debug,
        "MetalBackend::load_function_pointers_into_handle - Loaded AOTInductorModelContainerRun");

    LOAD_SYMBOL(
        handle,
        update_constants_from_blob,
        AOTInductorModelUpdateConstantsFromBlob,
        so_handle);
    ET_LOG(
        Debug,
        "MetalBackend::load_function_pointers_into_handle - Loaded AOTInductorModelUpdateConstantsFromBlob");

    ET_LOG(
        Debug,
        "MetalBackend::load_function_pointers_into_handle - All symbols loaded successfully");
    return Error::Ok;
  }

 public:
  // Once in program
  MetalBackend() {
    ET_LOG(Debug, "MetalBackend ctor");
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
    auto init_start = std::chrono::high_resolution_clock::now();
    ET_LOG(Info, "MetalBackend::init - Starting initialization");

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
    ET_LOG(Info, "MetalBackend::init - so_blob_key: %s", so_blob_key.c_str());

    const NamedDataMap* named_data_map = context.get_named_data_map();
    ET_LOG(Info, "MetalBackend::init - Got named data map: %p", named_data_map);

    ET_LOG(
        Info,
        "MetalBackend::init - Looking for blob key: %s",
        so_blob_key.c_str());

    auto aoti_metal_buffer = named_data_map->get_data(so_blob_key.c_str());
    ET_CHECK_OR_RETURN_ERROR(
        aoti_metal_buffer.ok(),
        Internal,
        "Failed to get data for key %s: 0x%x",
        so_blob_key.c_str(),
        static_cast<uint32_t>(aoti_metal_buffer.error()));

    ET_LOG(
        Info,
        "MetalBackend::init - Buffer is OK, size: %zu",
        aoti_metal_buffer->size());

    if (aoti_metal_buffer->data() == nullptr) {
      ET_LOG(Error, "MetalBackend::init - Buffer data is null");
      return Error::InvalidArgument;
    }

    ET_LOG(
        Info,
        "MetalBackend::init - Buffer data pointer: %p",
        aoti_metal_buffer->data());

    // Generate dynamic temporary file path
    filesystem::path temp_dir = filesystem::temp_directory_path();
    filesystem::path so_path =
        temp_dir / (so_blob_key + to_string(getpid()) + ".so");

    // Create a temporary file
    ET_LOG(
        Info, "MetalBackend::init - Creating temp file: %s", so_path.c_str());
    ofstream outfile(so_path.c_str(), ios::binary);

    // Write the ELF buffer to the temporary file
    ET_LOG(
        Info,
        "Writing %zu bytes to %s",
        aoti_metal_buffer->size(),
        so_path.c_str());

    outfile.write(
        static_cast<const char*>(aoti_metal_buffer->data()),
        aoti_metal_buffer->size());

    ET_CHECK_OR_RETURN_ERROR(
        outfile, AccessFailed, "Failed to write to file %s", so_path.c_str());

    // Finish writing the file to disk
    outfile.close();
    ET_LOG(Info, "MetalBackend::init - File closed successfully");

    // Free the buffer immediately after writing to disk
    aoti_metal_buffer->Free();

    // Load the ELF using dlopen
    void* so_handle = dlopen(so_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    ET_CHECK_OR_RETURN_ERROR(
        so_handle != nullptr,
        AccessFailed,
        "Failed to load shared library: %s",
        dlerror());

    processed->Free();

    // Create handle and load function pointers into it
    AOTIDelegateHandle* handle = new AOTIDelegateHandle();
    handle->so_handle = so_handle;
    handle->so_path = so_path.string();

    // Load function pointers specific to this handle's shared library
    ET_CHECK_OK_OR_RETURN_ERROR(
        load_function_pointers_into_handle(so_handle, handle));

    AOTInductorModelContainerHandle container_handle = nullptr;
    ET_LOG(
        Info,
        "MetalBackend::init - About to create AOTI container with device='mps'");

    ET_CHECK_OK_OR_RETURN_ERROR(
        handle->create_with_device(&container_handle, 1, "mps", nullptr));

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

    ET_LOG(Info, "MetalBackend::init - Initialization completed successfully");

    // Accumulate init timing statistics
    auto init_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(init_end - init_start)
            .count();
    g_init_total_ms += elapsed_ms;
    g_init_call_count++;

    // Track per-method init timing
    if (!method_name.empty()) {
      auto& stats = g_init_method_stats[method_name];
      stats.total_ms += elapsed_ms;
      stats.call_count++;
    }

    return (DelegateHandle*)handle; // Return the handle post-processing
  }

  // Once per execution
  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* handle_,
      Span<EValue*> args) const override {
    auto execute_start = std::chrono::high_resolution_clock::now();
    ET_LOG(Debug, "MetalBackend execute");

    AOTIDelegateHandle* handle = (AOTIDelegateHandle*)handle_;

    ET_LOG(Debug, "MetalBackend Handle generated");

    size_t n_inputs;
    handle->get_num_inputs(handle->container_handle, &n_inputs);

    size_t n_outputs;
    handle->get_num_outputs(handle->container_handle, &n_outputs);

    ET_LOG(Debug, "MetalBackend n_outputs %zd generated", n_outputs);

    ET_CHECK_OR_RETURN_ERROR(
        n_inputs + n_outputs == args.size(),
        InvalidArgument,
        "number of user input %zd and output %zd generated from AOT Inductor does not match ET runner's %zd. Exit.",
        n_inputs,
        n_outputs,
        args.size())

    ET_LOG(
        Debug,
        "number of user input %zd and output %zd generated from AOT Inductor matches ET runner's %zd.",
        n_inputs,
        n_outputs,
        args.size());

    int32_t mps_device_type = aoti_torch_device_type_mps(); // Returns 13

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
      ET_LOG(
          Debug,
          "MetalBackend input %d scalar_type=%d",
          i,
          static_cast<int32_t>(scalar_type));

      // Create GPU tensor with same shape
      std::vector<int64_t> sizes_vec(sizes.begin(), sizes.end());

      AOTITensorHandle gpu_input_handle;
      Error create_err = aoti_torch_empty_strided(
          sizes_vec.size(),
          sizes_vec.data(),
          nullptr, // use default strides
          static_cast<int32_t>(scalar_type),
          mps_device_type, // device_type = mps
          0, // device_index = 0
          &gpu_input_handle);

      if (create_err != Error::Ok) {
        ET_LOG(Error, "Failed to create GPU tensor for input %d", i);
        return Error::Internal;
      }

      // Log the created GPU tensor scalar type
      auto gpu_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(
          gpu_input_handle);
      ET_LOG(
          Debug,
          "MetalBackend created GPU tensor %d scalar_type=%d",
          i,
          static_cast<int32_t>(gpu_tensor->scalar_type()));

      gpu_inputs[i] = gpu_input_handle;

      // Log the CPU tensor data before copying to GPU
      void* cpu_data = cpu_tensor->mutable_data_ptr();
      if (cpu_data && cpu_tensor->numel() > 0) {
        float* cpu_float_data = (float*)cpu_data;
        ET_LOG(
            Debug,
            "CPU input %d data before copy: [%.3f, %.3f, %.3f, ...] (numel=%zd)",
            i,
            cpu_float_data[0],
            cpu_float_data[1],
            cpu_float_data[2],
            cpu_tensor->numel());
      }

      // Copy data from CPU to GPU
      Error copy_err = aoti_torch_copy_(gpu_inputs[i], cpu_tensor, 0);
      if (copy_err != Error::Ok) {
        ET_LOG(Error, "Failed to copy input %d from CPU to GPU", i);
        return Error::Internal;
      }

      // Log the GPU tensor scalar type after copy
      auto gpu_tensor_after =
          reinterpret_cast<executorch::runtime::etensor::Tensor*>(
              gpu_inputs[i]);
      ET_LOG(
          Debug,
          "MetalBackend GPU tensor %d scalar_type after copy=%d",
          i,
          static_cast<int32_t>(gpu_tensor_after->scalar_type()));

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
      ET_LOG(
          Debug,
          "MetalBackend output %d scalar_type=%d",
          i,
          static_cast<int32_t>(scalar_type));

      // Create GPU tensor with same shape for kernel output
      std::vector<int64_t> sizes_vec(sizes.begin(), sizes.end());

      AOTITensorHandle gpu_output_handle;
      Error create_err = aoti_torch_empty_strided(
          sizes_vec.size(),
          sizes_vec.data(),
          nullptr, // use default strides
          static_cast<int32_t>(scalar_type),
          mps_device_type, // device_type = mps
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
      ET_LOG(
          Debug,
          "  gpu_inputs[%d] = %p, data_ptr = %p",
          i,
          gpu_inputs[i],
          gpu_input_data);
    }
    for (int i = 0; i < n_outputs; i++) {
      void* gpu_output_data = gpu_outputs[i]->mutable_data_ptr();
      ET_LOG(
          Debug,
          "  gpu_outputs[%d] = %p, data_ptr = %p",
          i,
          gpu_outputs[i],
          gpu_output_data);
    }

    // Run AOTI container with GPU tensors
    AOTIRuntimeError error = handle->run(
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
    try {
      synchronize_metal_stream();
    } catch (const std::exception& e) {
      ET_LOG(
          Error,
          "Failed to synchronize Metal stream after kernel execution: %s",
          e.what());
      return Error::Internal;
    } catch (...) {
      ET_LOG(
          Error,
          "Failed to synchronize Metal stream after kernel execution: unknown exception");
      return Error::Internal;
    }

    ET_LOG(Debug, "MetalBackend running done and synchronized");

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

    // Accumulate timing statistics
    auto execute_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(execute_end - execute_start)
            .count();
    g_execute_total_ms += elapsed_ms;
    g_execute_call_count++;

    // Track per-method timing
    const char* method_name = context.get_method_name();
    if (method_name != nullptr) {
      auto& stats = g_method_stats[method_name];
      stats.total_ms += elapsed_ms;
      stats.call_count++;
    }

    return Error::Ok;
  }

  void destroy(DelegateHandle* handle_) const override {
    if (handle_ == nullptr) {
      return;
    }
    AOTIDelegateHandle* handle = (AOTIDelegateHandle*)handle_;

    // NOTE: AOTInductorModelContainerDelete does not work correctly with
    // multiple .so files. Deleting one container frees shared resources,
    // which causes segmentation faults when attempting to delete other
    // containers. As a workaround, we skip explicit container deletion
    // and defer cleanup to the OS.
    // TODO: Find a proper solution for safe container deletion.
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
      if (!remove_error) {
        ET_LOG(
            Info,
            "Removed temporary shared library file: %s",
            handle->so_path.c_str());
      }
    }

    delete handle;
    cleanup_memory();
    executorch::backends::aoti::cleanup_tensor_metadata();
    ET_LOG(Debug, "MetalBackend handle %p destroy", handle_);
  }
};

} // namespace executorch::backends::metal

namespace executorch::backends {
namespace {
auto cls = metal::MetalBackend();
executorch::runtime::Backend backend{"MetalBackend", &cls};
static executorch::runtime::Error success_with_compiler =
    register_backend(backend);
} // namespace
} // namespace executorch::backends
