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
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <cstdio>

#include <array>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
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
#include <executorch/extension/tensor/tensor_ptr_maker.h>

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

/**
 * Print SlimTensor debug information in a formatted style.
 *
 * Output format:
 * SlimTensor {
 *   data_ptr: 0x...
 *   sizes: [d0, d1, ...]
 *   strides: [s0, s1, ...]
 *   n_dim: X
 *   numel: Y
 *   dtype: TypeName
 * }
 */
void print_tensor(const SlimTensor* tensor, const char* name = nullptr) {
  if (tensor == nullptr) {
    ET_LOG(Info, "SlimTensor%s%s: nullptr", name ? " " : "", name ? name : "");
    return;
  }

  auto sizes = tensor->sizes();
  auto strides = tensor->strides();

  std::string sizes_str = "[";
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (i > 0)
      sizes_str += ", ";
    sizes_str += std::to_string(sizes[i]);
  }
  sizes_str += "]";

  std::string strides_str = "[";
  for (size_t i = 0; i < strides.size(); ++i) {
    if (i > 0)
      strides_str += ", ";
    strides_str += std::to_string(strides[i]);
  }
  strides_str += "]";

  ET_LOG(
      Info,
      "SlimTensor%s%s {\n"
      "  data_ptr: %p\n"
      "  sizes: %s\n"
      "  strides: %s\n"
      "  n_dim: %zu\n"
      "  numel: %zu\n"
      "  dtype: %s\n"
      "}",
      name ? " " : "",
      name ? name : "",
      tensor->data_ptr(),
      sizes_str.c_str(),
      strides_str.c_str(),
      tensor->dim(),
      tensor->numel(),
      slim::c10::toString(tensor->dtype()));
}

/**
 * Print ETensor (executorch::runtime::etensor::Tensor) debug information
 * in a formatted style.
 *
 * Output format:
 * ETensor {
 *   data_ptr: 0x...
 *   sizes: [d0, d1, ...]
 *   strides: [s0, s1, ...]
 *   n_dim: X
 *   numel: Y
 *   dtype: TypeName
 * }
 */
void print_tensor(const Tensor* tensor, const char* name = nullptr) {
  if (tensor == nullptr) {
    ET_LOG(Info, "ETensor%s%s: nullptr", name ? " " : "", name ? name : "");
    return;
  }

  auto sizes = tensor->sizes();
  auto strides = tensor->strides();

  std::string sizes_str = "[";
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (i > 0)
      sizes_str += ", ";
    sizes_str += std::to_string(sizes[i]);
  }
  sizes_str += "]";

  std::string strides_str = "[";
  for (size_t i = 0; i < strides.size(); ++i) {
    if (i > 0)
      strides_str += ", ";
    strides_str += std::to_string(strides[i]);
  }
  strides_str += "]";

  ET_LOG(
      Info,
      "ETensor%s%s {\n"
      "  data_ptr: %p\n"
      "  sizes: %s\n"
      "  strides: %s\n"
      "  n_dim: %zu\n"
      "  numel: %zu\n"
      "  dtype: %s\n"
      "}",
      name ? " " : "",
      name ? name : "",
      tensor->const_data_ptr(),
      sizes_str.c_str(),
      strides_str.c_str(),
      static_cast<size_t>(tensor->dim()),
      static_cast<size_t>(tensor->numel()),
      executorch::runtime::toString(tensor->scalar_type()));
}

} // anonymous namespace

class ET_EXPERIMENTAL CudaBackend final
    : public ::executorch::runtime::BackendInterface {
 private:
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
    return method_name == skip_copy_method_;
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
    ET_LOG(Info, "line 292");

    // executorch::backends::cuda::setCurrentCUDAStream(
    //   static_cast<cudaStream_t>(handle->cuda_stream),
    //   0  // device index
    // );

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
    ET_LOG(Info, "line 307");

    // NOTE: ExecuTorch tensors maybe on CPU or GPU due to the skip-copy
    // optimization We need to create GPU copies for CUDA kernel execution using
    // SlimTensor
    std::vector<SlimTensor*> gpu_inputs(n_inputs);
    std::vector<SlimTensor*> gpu_outputs(n_outputs);

    // Process input tensors: convert ETensor (CPU) to SlimTensor (GPU)
    for (size_t i = 0; i < n_inputs; i++) {
      auto* cpu_tensor = &(args[i]->toTensor());
      print_tensor(cpu_tensor, "cpu_tensor[0]");

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

          print_tensor(gpu_inputs[i], "gpu_input[0]");

          continue;
        }
      }

      // Data is on CPU - use from_etensor to copy to GPU
      gpu_inputs[i] =
          new SlimTensor(from_etensor(*cpu_tensor, CPU_DEVICE, DEFAULT_CUDA_DEVICE));
      print_tensor(gpu_inputs[i], "gpu_input[0]");
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

    ET_LOG(Info, "line 374");

    // Run AOTI container with GPU SlimTensors
    // NOTE: The AOTI model may REPLACE the output tensor pointers during run().
    // Our pre-allocated tensors might be deleted by the model, and gpu_outputs
    // will contain pointers to NEW tensors that the model allocated.
    AOTIRuntimeError error = handle->run(
        handle->container_handle,
        reinterpret_cast<Tensor**>(gpu_inputs.data()),
        n_inputs,
        reinterpret_cast<Tensor**>(gpu_outputs.data()),
        n_outputs,
        handle->cuda_stream,
        nullptr);

    ET_LOG(Info, "line 387");

    ET_CHECK_OR_RETURN_ERROR(
        error == Error::Ok,
        Internal,
        "AOTInductorModelContainerRun failed with error code %d",
        error);

    print_tensor(gpu_outputs[0], "gpu_output[0]");

    // Synchronize CUDA stream to ensure all GPU operations are complete
    // before reading output tensor metadata and copying data back to CPU.
    // Without this, the GPU operations are asynchronous and the output
    // tensor data/metadata may not be ready yet.
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(handle->cuda_stream);
    cudaError_t sync_err = cudaStreamSynchronize(cuda_stream);
    ET_CHECK_OR_RETURN_ERROR(
        sync_err == cudaSuccess,
        Internal,
        "Failed to synchronize CUDA stream: %s",
        cudaGetErrorString(sync_err));

    const bool copy_outputs = !should_skip_copy_for_method(handle->method_name);

    ET_LOG(Info, "line 398");

    if (copy_outputs) {
      ET_LOG(Info, "copy_outputs = true -- copying outputs back to CPU");
      // Copy GPU SlimTensor results back to CPU ETensors
      for (size_t i = 0; i < n_outputs; i++) {
        auto* cpu_output_tensor = &(args[i + n_inputs]->toTensor());
        ET_CHECK_OK_OR_RETURN_ERROR(
            copy_slimtensor_to_etensor(gpu_outputs[i], cpu_output_tensor),
            "Failed to copy GPU output %zu back to CPU ETensor",
            i);
      }
    } else {
      ET_LOG(Info, "copy_outputs = false -- keep gpu tensor on gpu");
      // Skip-copy optimization: wrap GPU data as ETensor using from_blob
      // The caller is responsible for handling GPU data directly
      //
      // IMPORTANT: The AOTI model may replace the output tensor pointers during
      // handle->run(). The tensors we pre-allocated might have been deleted by
      // the model, and gpu_outputs now contains pointers to NEW tensors that
      // the model allocated. We store these NEW tensors for lifetime management.
      {
        std::lock_guard<std::mutex> guard(cached_outputs_mutex_);
        auto& cached_outputs = cached_outputs_[handle];
        auto& cached_tensor_ptrs = cached_tensor_ptrs_[handle];

        // Delete the PREVIOUS round's tensors (allocated by AOTI model in the
        // previous run). We must delete them because the AOTI model expects us
        // to manage lifetimes of outputs it returns.
        for (auto* tensor : cached_outputs) {
          if (tensor != nullptr) {
            delete tensor;
          }
        }
        cached_outputs.clear();
        cached_tensor_ptrs.clear();

        for (size_t i = 0; i < n_outputs; i++) {
          // gpu_outputs[i] now points to a tensor allocated by the AOTI model
          // (it may have replaced our pre-allocated tensor during handle->run).
          // Store this pointer for lifetime management.
          cached_outputs.push_back(gpu_outputs[i]);

          print_tensor(cached_outputs[i], "cached_outputs[0]");

          // Create an ETensor wrapper pointing to the GPU data
          SlimTensor* cached = cached_outputs.back();
          auto slim_sizes = cached->sizes();
          auto slim_strides = cached->strides();

          std::vector<executorch::aten::SizesType> et_sizes(cached->dim());
          std::vector<executorch::aten::StridesType> et_strides(cached->dim());
          for (size_t d = 0; d < cached->dim(); d++) {
            et_sizes[d] =
                static_cast<executorch::aten::SizesType>(slim_sizes[d]);
            et_strides[d] =
                static_cast<executorch::aten::StridesType>(slim_strides[d]);
          }

          // Create TensorPtr wrapper - MUST be stored to keep TensorImpl alive!
          // The TensorImpl owns the sizes/strides arrays. If TensorPtr is
          // destroyed, the ETensor in args will have dangling pointers.
          auto tensor_ptr = executorch::extension::from_blob(
              cached->data_ptr(),
              std::move(et_sizes),
              std::move(et_strides),
              static_cast<executorch::aten::ScalarType>(cached->dtype()));

          // Assign the wrapped tensor to the output EValue
          args[i + n_inputs]->toTensor() = *tensor_ptr;

          print_tensor(&args[i + n_inputs]->toTensor(), "args[i + n_inputs]->toTensor()");

          // Store TensorPtr to keep TensorImpl alive until next execution
          cached_tensor_ptrs.push_back(std::move(tensor_ptr));
        }
      }
    }

    ET_LOG(Info, "line 451");

    return Error::Ok;
  }

  void destroy(DelegateHandle* handle_) const override {
    if (handle_ == nullptr) {
      return;
    }
    AOTIDelegateHandle* handle = (AOTIDelegateHandle*)handle_;

    // Clean up cached output tensors and TensorPtrs for this handle
    {
      std::lock_guard<std::mutex> guard(cached_outputs_mutex_);
      auto it = cached_outputs_.find(handle);
      if (it != cached_outputs_.end()) {
        for (auto* tensor : it->second) {
          if (tensor != nullptr) {
            delete tensor;
          }
        }
        cached_outputs_.erase(it);
      }
      // Also clean up cached TensorPtrs (they will be destroyed automatically
      // when erased, releasing the TensorImpl ownership)
      cached_tensor_ptrs_.erase(handle);
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
  // When copy-skip is enabled, output SlimTensors are cached here to keep
  // GPU memory alive while the caller processes the results.
  // Maps from AOTIDelegateHandle* to its cached outputs.
  mutable std::mutex cached_outputs_mutex_;
  mutable std::unordered_map<AOTIDelegateHandle*, std::vector<SlimTensor*>>
      cached_outputs_;
  // TensorPtr wrappers must be kept alive so the ETensor's TensorImpl
  // (which owns sizes/strides arrays) isn't destroyed when TensorPtr goes
  // out of scope. Store them alongside cached SlimTensors.
  mutable std::unordered_map<
      AOTIDelegateHandle*,
      std::vector<executorch::extension::TensorPtr>>
      cached_tensor_ptrs_;
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
