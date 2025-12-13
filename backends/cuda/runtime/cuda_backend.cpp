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

// Structure to hold a reference to a GPU tensor for "keep on device"
// optimization. Owns the tensor handle - must be deleted when no longer needed.
struct GpuTensorRef {
  AOTITensorHandle handle; // Tensor handle (owned, for later deletion)
  void* data_ptr; // GPU memory pointer (for D2D copy)
  size_t size_bytes; // Total size in bytes
};

class ET_EXPERIMENTAL CudaBackend final
    : public ::executorch::runtime::BackendInterface {
 private:
  // ============================================================================
  // GPU Tensor Storage for D2D Copy Optimization
  // ============================================================================
  //
  // This backend supports storing GPU tensors between execute() calls to enable
  // device-to-device (D2D) copies instead of slower host-to-device (H2D)
  // copies. This is useful for encoder-decoder models where the encoder output
  // is reused across many decoder iterations.
  //
  // SUPPORTED OPTIONS (via set_option):
  //
  //   "store_output" (string): Store the output tensor under this name after
  //       the next execute() call. The tensor remains on GPU until cleared.
  //       Only supports single-output methods.
  //       Example: opts.set_option("store_output", "encoder_output");
  //
  //   "use_stored_input" (string): For inputs matching the stored tensor's
  //   size,
  //       use D2D copy from the stored tensor instead of H2D copy from CPU.
  //       This setting persists across execute() calls until reset.
  //       Example: opts.set_option("use_stored_input", "encoder_output");
  //
  //   "reset_stored_input" (bool): Clear the use_stored_input setting.
  //       Does NOT delete the stored tensor - only stops using it for D2D.
  //       Example: opts.set_option("reset_stored_input", true);
  //
  //   "clear_stored_tensor" (string): Delete the named tensor from storage,
  //       freeing GPU memory. Use after decoder loop completes.
  //       Example: opts.set_option("clear_stored_tensor", "encoder_output");
  //
  // TYPICAL USAGE PATTERN (encoder-decoder model):
  //
  //   1. Before encoder: set_option("store_output", "encoder_output")
  //   2. Execute encoder (output is stored on GPU)
  //   3. Before decoder loop: set_option("use_stored_input", "encoder_output")
  //   4. Execute decoder N times (D2D copies for encoder output input)
  //   5. After decoder loop:
  //        set_option("reset_stored_input", true)
  //        set_option("clear_stored_tensor", "encoder_output")
  //
  // ============================================================================

  // Storage control options (set via set_option before execute)
  mutable std::string
      store_output_name_; // Name to store output under (empty = none)
  mutable std::string
      use_stored_input_name_; // Name of stored tensor to use (empty = none)

  // Per-instance map of named GPU tensor references.
  // Mutable because execute() is const but needs to modify this.
  //
  // LIFETIME CONTRACT:
  // - Stored tensors are valid until overwritten or destroy() is called.
  // - Caller must ensure the producing execute() call (e.g., encoder) completes
  //   before any consuming execute() call (e.g., decoder) begins.
  // - Caller must not call destroy() while execute() is in progress.
  // - Overwriting a tensor (same name) deletes the old tensor immediately,
  //   so caller must ensure no concurrent execute() is using it.
  mutable std::unordered_map<std::string, GpuTensorRef> gpu_tensors_;

  // Helper to clear stored GPU tensors and free their memory.
  // Only call when no execute() is in progress.
  void clear_gpu_tensors() const {
    for (auto& pair : gpu_tensors_) {
      if (pair.second.handle != nullptr) {
        aoti_torch_delete_tensor_object(pair.second.handle);
      }
    }
    gpu_tensors_.clear();
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
      __ET_UNUSED executorch::runtime::BackendOptionContext& context,
      const executorch::runtime::Span<executorch::runtime::BackendOption>&
          backend_options) override {
    for (size_t i = 0; i < backend_options.size(); i++) {
      const auto& option = backend_options[i];
      // Handle store_output: expects a string name (e.g., "encoder_output")
      if (strcmp(option.key, "store_output") == 0) {
        if (auto* arr = std::get_if<
                std::array<char, executorch::runtime::kMaxOptionValueLength>>(
                &option.value)) {
          store_output_name_ = std::string(arr->data());
        } else {
          ET_LOG(Error, "store_output option expects a string value");
          return Error::InvalidArgument;
        }
      }
      // Handle use_stored_input: expects a string name (e.g., "encoder_output")
      else if (strcmp(option.key, "use_stored_input") == 0) {
        if (auto* arr = std::get_if<
                std::array<char, executorch::runtime::kMaxOptionValueLength>>(
                &option.value)) {
          use_stored_input_name_ = std::string(arr->data());
        } else {
          ET_LOG(Error, "use_stored_input option expects a string value");
          return Error::InvalidArgument;
        }
      }
      // Handle reset_stored_input: expects a boolean value
      // Note: This only resets the name setting. The stored GPU tensor
      // remains in memory until overwritten or destroy() is called.
      else if (strcmp(option.key, "reset_stored_input") == 0) {
        if (auto* val = std::get_if<bool>(&option.value)) {
          if (*val) {
            use_stored_input_name_.clear();
          }
        } else {
          ET_LOG(Error, "reset_stored_input option expects a boolean value");
          return Error::InvalidArgument;
        }
      }
      // Handle clear_stored_tensor: expects a string name
      // Deletes the named GPU tensor from storage, freeing GPU memory.
      else if (strcmp(option.key, "clear_stored_tensor") == 0) {
        if (auto* arr = std::get_if<
                std::array<char, executorch::runtime::kMaxOptionValueLength>>(
                &option.value)) {
          std::string name(arr->data());
          auto it = gpu_tensors_.find(name);
          if (it != gpu_tensors_.end()) {
            if (it->second.handle != nullptr) {
              aoti_torch_delete_tensor_object(it->second.handle);
            }
            gpu_tensors_.erase(it);
          }
        } else {
          ET_LOG(Error, "clear_stored_tensor option expects a string value");
          return Error::InvalidArgument;
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

    // RAII helper to ensure GPU tensors are cleaned up on all exit paths.
    // Prevents memory leaks when errors occur during execute().
    struct TensorCleanup {
      std::vector<AOTITensorHandle>& inputs;
      std::vector<AOTITensorHandle>& outputs;
      const std::unordered_map<std::string, GpuTensorRef>& stored_tensors;

      ~TensorCleanup() {
        // Clean up input tensors
        for (auto* handle : inputs) {
          if (handle != nullptr) {
            aoti_torch_delete_tensor_object(handle);
          }
        }
        // Clean up output tensors, except those that are stored
        for (auto* handle : outputs) {
          if (handle != nullptr) {
            bool is_stored = false;
            for (const auto& pair : stored_tensors) {
              if (pair.second.handle == handle) {
                is_stored = true;
                break;
              }
            }
            if (!is_stored) {
              aoti_torch_delete_tensor_object(handle);
            }
          }
        }
      }
    };
    TensorCleanup cleanup{gpu_inputs, gpu_outputs, gpu_tensors_};

    // Track which input index was matched for D2D copy (for duplicate
    // detection)
    ssize_t matched_input_idx = -1;

    // Process input tensors: ExecuTorch provides CPU tensors, create GPU
    // copies. For stored inputs, use GPU-to-GPU copy instead of CPU-to-GPU.
    for (size_t i = 0; i < n_inputs; i++) {
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
          "Failed to create GPU tensor for input %zu",
          i);

      gpu_inputs[i] = gpu_input_handle;

      // Check if this input matches a stored GPU tensor (by size).
      if (!use_stored_input_name_.empty()) {
        auto it = gpu_tensors_.find(use_stored_input_name_);
        if (it != gpu_tensors_.end()) {
          const GpuTensorRef& ref = it->second;
          size_t numel = gpu_inputs[i]->numel();
          size_t elem_size = gpu_inputs[i]->element_size();
          size_t copy_bytes = numel * elem_size;

          // Match by size: use stored tensor if sizes match
          if (copy_bytes == ref.size_bytes) {
            if (matched_input_idx >= 0) {
              // Another input already matched - warn about ambiguity
              ET_LOG(
                  Error,
                  "Multiple inputs match stored tensor '%s' size (%zu bytes): "
                  "input %zd was used, input %zu also matches. "
                  "Consider using unique tensor sizes or a different matching strategy.",
                  use_stored_input_name_.c_str(),
                  copy_bytes,
                  matched_input_idx,
                  i);
            } else {
              // First match - perform D2D copy
              matched_input_idx = static_cast<ssize_t>(i);

              ET_LOG(
                  Debug,
                  "Using stored tensor '%s' for input %zu (%zu bytes, D2D copy)",
                  use_stored_input_name_.c_str(),
                  i,
                  copy_bytes);

              // GPU-to-GPU copy: fast DMA transfer, normalizes tensor format
              cudaError_t cuda_err = cudaMemcpy(
                  gpu_inputs[i]->data_ptr(),
                  ref.data_ptr,
                  copy_bytes,
                  cudaMemcpyDeviceToDevice);

              ET_CHECK_OR_RETURN_ERROR(
                  cuda_err == cudaSuccess,
                  Internal,
                  "Failed GPU-to-GPU copy for input %zu: %s",
                  i,
                  cudaGetErrorString(cuda_err));

              // Skip the CPU-to-GPU copy below
              continue;
            }
          }
        }
      }

      // Copy data from CPU to GPU (normal path)
      ET_CHECK_OR_RETURN_ERROR(
          aoti_torch_copy_(gpu_inputs[i], cpu_tensor, 0) == Error::Ok,
          Internal,
          "Failed to copy input %zu from CPU to GPU",
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
          "Failed to create GPU tensor for output %zu",
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

    // Store reference to output GPU tensor if requested.
    // The tensor will be kept alive for later D2D copy to decoder inputs.
    if (!store_output_name_.empty()) {
      ET_CHECK_OR_RETURN_ERROR(
          n_outputs == 1,
          InvalidArgument,
          "store_output only supports single-output methods, got %zu outputs",
          n_outputs);

      auto* gpu_tensor = gpu_outputs[0];
      size_t numel = gpu_tensor->numel();
      size_t elem_size = gpu_tensor->element_size();
      size_t size_bytes = numel * elem_size;

      // Delete old tensor if overwriting (erase first to prevent double-free)
      auto old_it = gpu_tensors_.find(store_output_name_);
      if (old_it != gpu_tensors_.end()) {
        AOTITensorHandle old_handle = old_it->second.handle;
        gpu_tensors_.erase(old_it); // Remove from map before deleting
        if (old_handle != nullptr) {
          aoti_torch_delete_tensor_object(old_handle);
        }
      }

      // Store tensor reference (we now own this tensor)
      GpuTensorRef ref;
      ref.handle = gpu_tensor;
      ref.data_ptr = gpu_tensor->data_ptr();
      ref.size_bytes = size_bytes;
      gpu_tensors_[store_output_name_] = ref;

      // Reset store_output name after storing
      store_output_name_.clear();
    }

    // Copy GPU output results back to CPU output tensors
    for (size_t i = 0; i < n_outputs; i++) {
      auto cpu_output_tensor = &(args[i + n_inputs]->toTensor());
      // For DYNAMIC_BOUND tensors we try to resize
      ET_CHECK_OK_OR_RETURN_ERROR(
          resize_tensor(*cpu_output_tensor, gpu_outputs[i]->sizes()),
          "Error resizing tensor at output index %zu",
          i);
      ET_CHECK_OK_OR_RETURN_ERROR(
          aoti_torch_copy_(cpu_output_tensor, gpu_outputs[i], 0),
          "Failed to copy GPU output %zu back to CPU",
          i);
    }

    // Memory management notes:
    // - GPU tensor cleanup is handled by TensorCleanup RAII guard above.
    // - use_stored_input setting persists across execute() calls to support
    //   decoder loops that reuse the stored encoder output.
    // - Stored GPU tensors (in gpu_tensors_) remain in memory until:
    //   (a) overwritten by a new tensor with the same name, or
    //   (b) destroy() is called, which frees all stored tensors.
    // - The "reset_stored_input" option only resets the input name setting,
    //   NOT the stored GPU tensors themselves.

    return Error::Ok;
  }

  void destroy(DelegateHandle* handle_) const override {
    if (handle_ == nullptr) {
      return;
    }
    AOTIDelegateHandle* handle = (AOTIDelegateHandle*)handle_;

    // Delete stored GPU tensors
    clear_gpu_tensors();

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
