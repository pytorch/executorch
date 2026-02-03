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
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <cctype>
#include <cstdio>

#include <array>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Include SlimTensor headers for CUDA backend
#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/cuda/Exception.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/aoti/slim/core/storage.h>
#include <executorch/backends/aoti/slim/factory/empty.h>
#include <executorch/backends/aoti/slim/factory/from_blob.h>
#include <executorch/backends/aoti/slim/factory/from_etensor.h>
#include <executorch/backends/aoti/slim/util/array_ref_util.h>

// Include our shim layer headers
#include <executorch/backends/aoti/aoti_delegate_handle.h>
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

// SlimTensor type aliases
using slim::CPU_DEVICE;
using slim::DEFAULT_CUDA_DEVICE;
using slim::DeviceTraits;
using slim::from_etensor;
using slim::SlimTensor;
using slim::c10::Device;
using slim::c10::DeviceType;

namespace {
constexpr char kSkipCopyOutputToCpuForMethod[] =
    "skip_copy_output_to_cpu_for_method";
} // anonymous namespace

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

    // NOTE: ExecuTorch tensors may be on CPU or GPU due to the skip-copy
    // optimization. We need to create GPU copies for CUDA kernel execution
    // using SlimTensor.
    std::vector<SlimTensor*> gpu_inputs(n_inputs);
    std::vector<SlimTensor*> gpu_outputs(n_outputs);

    // Process input tensors: convert ETensor (CPU) to SlimTensor (GPU)
    for (size_t i = 0; i < n_inputs; i++) {
      auto* cpu_tensor = &(args[i]->toTensor());

      // Check if input data is already on GPU (skip-copy optimization for
      // inputs) This can happen when the caller has pre-staged data on GPU
      cudaPointerAttributes attributes{};
      const void* data_ptr = cpu_tensor->const_data_ptr();
      if (data_ptr != nullptr) {
        cudaError_t err = cudaPointerGetAttributes(&attributes, data_ptr);
        if (err == cudaSuccess && attributes.type == cudaMemoryTypeDevice) {
          // Data is already on GPU - wrap it directly without copy
          auto sizes = cpu_tensor->sizes();
          auto strides = cpu_tensor->strides();
          std::vector<int64_t> sizes_vec(sizes.begin(), sizes.end());
          std::vector<int64_t> strides_vec(strides.begin(), strides.end());

          gpu_inputs[i] = new SlimTensor(slim::from_blob(
              const_cast<void*>(data_ptr),
              slim::makeArrayRef(sizes_vec),
              slim::makeArrayRef(strides_vec),
              static_cast<slim::c10::ScalarType>(cpu_tensor->scalar_type()),
              DEFAULT_CUDA_DEVICE,
              0 // storage_offset
              ));

          continue;
        }
      }

      // Data is on CPU - use from_etensor to copy to GPU
      gpu_inputs[i] = new SlimTensor(
          from_etensor(*cpu_tensor, CPU_DEVICE, DEFAULT_CUDA_DEVICE));
    }

    // Process output tensors: create GPU SlimTensors for kernel output
    for (size_t i = 0; i < n_outputs; i++) {
      auto* cpu_output_tensor = &(args[i + n_inputs]->toTensor());
      auto sizes = cpu_output_tensor->sizes();
      auto strides = cpu_output_tensor->strides();
      auto scalar_type = cpu_output_tensor->scalar_type();

      std::vector<int64_t> sizes_vec(sizes.begin(), sizes.end());
      std::vector<int64_t> strides_vec(strides.begin(), strides.end());

      gpu_outputs[i] = new SlimTensor(slim::empty_strided(
          slim::makeArrayRef(sizes_vec),
          slim::makeArrayRef(strides_vec),
          static_cast<slim::c10::ScalarType>(scalar_type),
          DEFAULT_CUDA_DEVICE));
    }

    // Run the AOTI container with SlimTensors.
    //
    // NOTE: The handle->run function (defined in aoti_delegate_handle.h)
    // expects ETensor* as input/output. We avoid changing its signature since
    // it's shared with the Metal backend. Instead, we reinterpret_cast
    // SlimTensor* to Tensor*
    AOTIRuntimeError error = handle->run(
        handle->container_handle,
        reinterpret_cast<Tensor**>(gpu_inputs.data()),
        n_inputs,
        reinterpret_cast<Tensor**>(gpu_outputs.data()),
        n_outputs,
        handle->cuda_stream,
        nullptr);

    ET_CHECK_OR_RETURN_ERROR(
        error == Error::Ok,
        Internal,
        "AOTInductorModelContainerRun failed with error code %d",
        error);

    const bool copy_outputs = !should_skip_copy_for_method(handle->method_name);

    // // Synchronize CUDA stream to ensure kernel execution is complete
    // // before accessing output data (either for copy or skip-copy path)
    // cudaStream_t cuda_stream = static_cast<cudaStream_t>(handle->cuda_stream);
    // cudaError_t sync_err = cudaStreamSynchronize(cuda_stream);
    // ET_CHECK_OR_RETURN_ERROR(
    //     sync_err == cudaSuccess,
    //     Internal,
    //     "cudaStreamSynchronize failed: %s",
    //     cudaGetErrorString(sync_err));

    if (copy_outputs) {
      // Deep copy GPU SlimTensor results back to CPU ETensors
      for (size_t i = 0; i < n_outputs; i++) {
        auto* cpu_output_tensor = &(args[i + n_inputs]->toTensor());
        ET_CHECK_OK_OR_RETURN_ERROR(
            copy_slimtensor_to_etensor(gpu_outputs[i], cpu_output_tensor),
            "Failed to copy GPU output %zu back to CPU ETensor",
            i);
      }
      // Cleanup gpu_outputs after copying - they are no longer needed
      delete_slimtensor_vector(gpu_outputs);
    } else {
      // Skip-copy optimization: point ETensor directly to GPU data.
      // The caller is responsible for handling GPU data directly.
      //
      // Lifetime management: We cache the newly created GPU tensors and delete
      // the previous round's tensors, since they are no longer needed.
      {
        std::lock_guard<std::mutex> guard(cached_outputs_mutex_);
        auto& cached_outputs = cached_outputs_[handle];

        // Delete the previous round's tensors since they are no longer in use.
        delete_slimtensor_vector(cached_outputs);

        for (size_t i = 0; i < n_outputs; i++) {
          // Cache this output tensor to keep the underlying GPU data alive.
          cached_outputs.push_back(gpu_outputs[i]);

          // Wrap the GPU SlimTensor data into the ETensor (zero-copy).
          // This resizes the ETensor to match the SlimTensor shape and sets
          // its data pointer to point directly to the GPU data.
          auto* output_etensor = &(args[i + n_inputs]->toTensor());
          ET_CHECK_OK_OR_RETURN_ERROR(
              wrap_slimtensor_to_etensor(gpu_outputs[i], output_etensor),
              "Failed to wrap GPU output %zu into ETensor",
              i);
        }
      }
    }

    // Cleanup gpu_inputs - they are no longer needed after kernel execution
    delete_slimtensor_vector(gpu_inputs);

    return Error::Ok;
  }

  void destroy(DelegateHandle* handle_) const override {
    if (handle_ == nullptr) {
      return;
    }
    AOTIDelegateHandle* handle = (AOTIDelegateHandle*)handle_;

    // Clean up cached output tensors for this handle
    {
      std::lock_guard<std::mutex> guard(cached_outputs_mutex_);
      auto it = cached_outputs_.find(handle);
      if (it != cached_outputs_.end()) {
        delete_slimtensor_vector(it->second);
        cached_outputs_.erase(it);
      }
    }

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
      Error err = close_library(handle->so_handle);
      ET_CHECK_OR_LOG_ERROR(
          err == Error::Ok,
          "Failed to close shared library for %s",
          handle->so_path.c_str());
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
  }

 private:
  mutable std::mutex skip_copy_method_mutex_;
  std::string skip_copy_method_;

  // Cached output tensors for skip-copy optimization.
  // When skip-copy is enabled, output SlimTensors are cached here to keep
  // the underlying GPU memory alive while the caller processes the results.
  // Maps each AOTIDelegateHandle* to its vector of cached output tensors.
  mutable std::mutex cached_outputs_mutex_;
  mutable std::unordered_map<AOTIDelegateHandle*, std::vector<SlimTensor*>>
      cached_outputs_;
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
