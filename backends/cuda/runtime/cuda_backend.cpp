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
#include <cctype>
#include <cstdio>

#include <array>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
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
using executorch::runtime::BackendOption;
using executorch::runtime::BackendOptionContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::kMaxOptionValueLength;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::NamedDataMap;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::etensor::Tensor;

namespace {
constexpr char kSkipCopyOutputToCpuForMethod[] =
    "skip_copy_output_to_cpu_for_method";
}

class ET_EXPERIMENTAL CudaBackend final
    : public ::executorch::runtime::BackendInterface {
 private:
  // Trim leading/trailing whitespace from a view of the string.
  static std::string_view trim(std::string_view s) {
    size_t start = 0;
    while (start < s.size() &&
           std::isspace(static_cast<unsigned char>(s[start]))) {
      ++start;
    }
    size_t end = s.size();
    while (end > start &&
           std::isspace(static_cast<unsigned char>(s[end - 1]))) {
      --end;
    }
    return s.substr(start, end - start);
  }

  // Check if method_name appears in a comma-separated list.
  static bool method_in_csv(
      const std::string& method_name,
      const std::string& csv) {
    size_t pos = 0;
    while (pos <= csv.size()) {
      const size_t comma = csv.find(',', pos);
      const std::string_view token =
          trim(std::string_view(csv).substr(pos, comma - pos));
      if (!token.empty() && token == method_name) {
        return true;
      }
      if (comma == std::string::npos) {
        break;
      }
      pos = comma + 1;
    }
    return false;
  }

  void set_skip_copy_method(
      const std::array<char, kMaxOptionValueLength>& raw) {
    std::lock_guard<std::mutex> guard(skip_copy_method_mutex_);
    skip_copy_method_ = std::string(raw.data());
  }

  std::array<char, kMaxOptionValueLength> get_skip_copy_method_as_option()
      const {
    std::array<char, kMaxOptionValueLength> out{};
    std::string value;
    {
      std::lock_guard<std::mutex> guard(skip_copy_method_mutex_);
      value = skip_copy_method_;
    }
    std::snprintf(out.data(), out.size(), "%s", value.c_str());
    return out;
  }

  bool should_skip_copy_for_method(const std::string& method_name) const {
    if (method_name.empty()) {
      return false;
    }
    std::lock_guard<std::mutex> guard(skip_copy_method_mutex_);
    return method_in_csv(method_name, skip_copy_method_);
  }

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
      ET_UNUSED BackendOptionContext& context,
      const executorch::runtime::Span<BackendOption>& backend_options)
      override {
    for (const auto& option : backend_options) {
      if (std::strcmp(option.key, kSkipCopyOutputToCpuForMethod) == 0) {
        if (auto* val = std::get_if<std::array<char, kMaxOptionValueLength>>(
                &option.value)) {
          set_skip_copy_method(*val);
        } else {
          ET_LOG(
              Error,
              "Option %s must be a method name string.",
              kSkipCopyOutputToCpuForMethod);
          return Error::InvalidArgument;
        }
      }
    }
    return Error::Ok;
  }

  Error get_option(
      ET_UNUSED BackendOptionContext& context,
      executorch::runtime::Span<BackendOption>& backend_options) override {
    for (auto& option : backend_options) {
      if (std::strcmp(option.key, kSkipCopyOutputToCpuForMethod) == 0) {
        option.value = get_skip_copy_method_as_option();
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
    handle->method_name = method_name;

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
    // copies
    for (size_t i = 0; i < n_inputs; i++) {
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
          "Failed to create GPU tensor for input %" ET_PRIsize_t,
          i);

      gpu_inputs[i] = gpu_input_handle;

      // Copy data from CPU to GPU
      ET_CHECK_OR_RETURN_ERROR(
          aoti_torch_copy_(gpu_inputs[i], cpu_tensor, 0) == Error::Ok,
          Internal,
          "Failed to copy input %" ET_PRIsize_t " from CPU to GPU",
          i);
    }
    // Process output tensors: create GPU counterparts for ExecuTorch CPU
    // tensors
    for (size_t i = 0; i < n_outputs; i++) {
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
          "Failed to create GPU tensor for output %" ET_PRIsize_t,
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

    const bool copy_outputs = !should_skip_copy_for_method(handle->method_name);

    if (copy_outputs) {
      // Copy GPU output results back to CPU output tensors
      for (size_t i = 0; i < n_outputs; i++) {
        auto cpu_output_tensor = &(args[i + n_inputs]->toTensor());
        // For DYNAMIC_BOUND tensors we try to resize
        ET_CHECK_OK_OR_RETURN_ERROR(
            resize_tensor(*cpu_output_tensor, gpu_outputs[i]->sizes()),
            "Error resizing tensor at output index %" ET_PRIsize_t,
            i);
        ET_CHECK_OK_OR_RETURN_ERROR(
            aoti_torch_copy_(cpu_output_tensor, gpu_outputs[i], 0),
            "Failed to copy GPU output %" ET_PRIsize_t " back to CPU",
            i);
      }
    } else {
      for (size_t i = 0; i < n_outputs; i++) {
        args[i + n_inputs]->toTensor() = *gpu_outputs[i];
      }
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

 private:
  mutable std::mutex skip_copy_method_mutex_;
  std::string skip_copy_method_;
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
