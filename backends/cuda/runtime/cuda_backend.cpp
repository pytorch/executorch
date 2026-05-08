/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/safe_numerics.h>
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
#include <atomic>
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
#include <executorch/backends/aoti/utils.h>
#include <executorch/backends/cuda/runtime/cuda_delegate_handle.h>
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
using cuda::CudaGraphPhase;
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
constexpr char kUseSharedCudaStream[] = "use_shared_cuda_stream";

constexpr char kEnableCudaGraphForMethod[] = "enable_cuda_graph_for_method";
constexpr int kCudaGraphWarmupSteps = 3;
constexpr char kWeightSharingAcrossMethods[] = "weight_sharing_across_methods";
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

  void set_cuda_graph_method(
      const std::array<char, kMaxOptionValueLength>& raw) {
    std::lock_guard<std::mutex> guard(cuda_graph_method_mutex_);
    cuda_graph_method_ = std::string(raw.data());
  }

  bool should_use_cuda_graph_for_method(const std::string& method_name) const {
    if (method_name.empty()) {
      return false;
    }
    std::lock_guard<std::mutex> guard(cuda_graph_method_mutex_);
    return method_in_csv(method_name, cuda_graph_method_);
  }

  // Create the shared CUDA stream. Called when use_shared_cuda_stream option
  // is set to true. The presence of shared_cuda_stream_ indicates shared mode.
  void create_shared_cuda_stream() {
    std::lock_guard<std::mutex> guard(cuda_stream_mutex_);
    if (shared_cuda_stream_ != nullptr) {
      return; // Already created
    }
    shared_cuda_stream_ = cuda::create_cuda_stream();
    if (shared_cuda_stream_ == nullptr) {
      ET_LOG(Error, "Failed to create shared CUDA stream");
      return;
    }
    ET_LOG(Info, "Created shared CUDA stream: %p", *shared_cuda_stream_);
  }

  // Get the shared CUDA stream. Returns nullptr if not in shared mode.
  std::shared_ptr<cudaStream_t> get_shared_cuda_stream() const {
    std::lock_guard<std::mutex> guard(cuda_stream_mutex_);
    return shared_cuda_stream_;
  }

  // Check if we're using shared CUDA stream mode.
  bool is_using_shared_cuda_stream() const {
    std::lock_guard<std::mutex> guard(cuda_stream_mutex_);
    return shared_cuda_stream_ != nullptr;
  }

  // Enable cross-method per-FQN weight caching. Set via the
  // kWeightSharingAcrossMethods runtime backend option.
  void set_weight_sharing_across_methods(bool enabled) {
    weight_sharing_across_methods_.store(enabled, std::memory_order_relaxed);
  }

  bool is_weight_sharing_across_methods_enabled() const {
    return weight_sharing_across_methods_.load(std::memory_order_relaxed);
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

    // Load constant management symbols (optional — needed for cross-method
    // buffer sharing). These are available in torch >= 2.6.
#define LOAD_OPTIONAL_SYMBOL(member, name)                            \
  do {                                                                \
    auto res = get_function(so_handle, #name);                        \
    handle->member =                                                  \
        res.ok() ? reinterpret_cast<name##Func>(res.get()) : nullptr; \
  } while (0)

    LOAD_OPTIONAL_SYMBOL(
        get_num_constants, AOTInductorModelContainerGetNumConstants);
    LOAD_OPTIONAL_SYMBOL(
        get_constant_name, AOTInductorModelContainerGetConstantName);
    LOAD_OPTIONAL_SYMBOL(
        get_constant_original_fqn,
        AOTInductorModelContainerGetConstantOriginalFQN);
    LOAD_OPTIONAL_SYMBOL(
        extract_constants_map, AOTInductorModelContainerExtractConstantsMap);
    LOAD_OPTIONAL_SYMBOL(
        update_user_managed_constant_buffer_pairs,
        AOTInductorModelContainerUpdateUserManagedConstantBufferPairs);
#undef LOAD_OPTIONAL_SYMBOL

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
      } else if (std::strcmp(option.key, kUseSharedCudaStream) == 0) {
        if (auto* val = std::get_if<bool>(&option.value)) {
          if (*val) {
            create_shared_cuda_stream();
          }
        } else {
          ET_LOG(Error, "Option %s must be a boolean.", kUseSharedCudaStream);
          return Error::InvalidArgument;
        }
      } else if (std::strcmp(option.key, kWeightSharingAcrossMethods) == 0) {
        if (auto* val = std::get_if<bool>(&option.value)) {
          set_weight_sharing_across_methods(*val);
        } else {
          ET_LOG(
              Error,
              "Option %s must be a boolean.",
              kWeightSharingAcrossMethods);
          return Error::InvalidArgument;
        }
      } else if (std::strcmp(option.key, kEnableCudaGraphForMethod) == 0) {
        if (auto* val = std::get_if<std::array<char, kMaxOptionValueLength>>(
                &option.value)) {
          set_cuda_graph_method(*val);
        } else {
          ET_LOG(
              Error,
              "Option %s must be a method name string.",
              kEnableCudaGraphForMethod);
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
    cuda::CudaDelegateHandle* handle = new cuda::CudaDelegateHandle();
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

    // Load constants. When weight_sharing_across_methods is enabled (opt-in
    // via the kWeightSharingAcrossMethods runtime backend option set by the
    // runner), use the per-weight FQN cache so methods that share weights
    // (e.g. prefill/decode) avoid duplicate GPU allocations. Otherwise fall
    // back to the legacy per-method blob load — required for models whose
    // methods are independent sub-graphs that may have FQN collisions
    // (e.g. parakeet).
    if (is_weight_sharing_across_methods_enabled()) {
      ET_CHECK_OK_OR_RETURN_ERROR(
          load_constants_with_cache(handle, named_data_map, method_name));
    } else {
      ET_CHECK_OK_OR_RETURN_ERROR(
          load_constants_legacy(handle, named_data_map, method_name));
    }

    // Use shared CUDA stream if enabled via options, otherwise create one.
    // A shared stream ensures proper ordering across multiple methods
    // (e.g., encoder, decoder, sampler) when using skip-copy optimization.
    if (is_using_shared_cuda_stream()) {
      // Shared stream mode: all handles share the same stream.
      handle->cuda_stream = get_shared_cuda_stream();
      ET_LOG(
          Info,
          "Using shared CUDA stream %p for method %s",
          handle->get_cuda_stream(),
          method_name.c_str());
    } else {
      // Per-handle stream mode: each handle owns its own stream.
      handle->cuda_stream = cuda::create_cuda_stream();
      if (handle->cuda_stream == nullptr) {
        delete handle;
        return Error::Internal;
      }
      ET_LOG(
          Info,
          "Created new CUDA stream %p for method %s",
          handle->get_cuda_stream(),
          method_name.c_str());
    }

    // Initialize CUDA graph state if enabled for this method.
    if (should_use_cuda_graph_for_method(method_name)) {
      handle->cuda_graph_state.phase = CudaGraphPhase::Warmup;
      handle->cuda_graph_state.warmup_remaining = kCudaGraphWarmupSteps;
      ET_LOG(
          Info,
          "CUDA graph enabled for method '%s' (warmup=%d)",
          method_name.c_str(),
          kCudaGraphWarmupSteps);
    }

    return (DelegateHandle*)handle; // Return the handle post-processing
  }

  // Once per execution
  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* handle_,
      Span<EValue*> args) const override {
    cuda::CudaDelegateHandle* handle = (cuda::CudaDelegateHandle*)handle_;

    size_t n_inputs;
    handle->get_num_inputs(handle->container_handle, &n_inputs);

    size_t n_outputs;
    handle->get_num_outputs(handle->container_handle, &n_outputs);

    setCurrentCUDAStream(handle->get_cuda_stream(), 0);

    size_t n_io_sum = 0;
    ET_CHECK_OR_RETURN_ERROR(
        !c10::add_overflows(n_inputs, n_outputs, &n_io_sum) &&
            n_io_sum == args.size(),
        InvalidArgument,
        "number of user input %zd and output %zd generated from AOT Inductor does not match ET runner's %zd. Exit.",
        n_inputs,
        n_outputs,
        args.size())

    // ---------------------------------------------------------------
    // CUDA graph REPLAY path — skip all tensor setup and just replay
    // ---------------------------------------------------------------
    if (handle->cuda_graph_state.phase == CudaGraphPhase::Replay) {
      Result<cudaStream_t> csr = getCurrentCUDAStream(0);
      ET_CHECK_OK_OR_RETURN_ERROR(csr.error());
      cudaStream_t cs = csr.get();

      // Copy new input data into static input buffers
      for (size_t i = 0; i < n_inputs; i++) {
        auto* cpu_tensor = &(args[i]->toTensor());
        ET_CHECK_OR_RETURN_ERROR(
            cpu_tensor->nbytes() ==
                handle->cuda_graph_state.static_input_nbytes[i],
            InvalidArgument,
            "CUDA graph replay: input %zu size mismatch (expected %zu, got %zu)",
            i,
            handle->cuda_graph_state.static_input_nbytes[i],
            cpu_tensor->nbytes());
        ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMemcpyAsync(
            handle->cuda_graph_state.static_input_ptrs[i],
            cpu_tensor->const_data_ptr(),
            handle->cuda_graph_state.static_input_nbytes[i],
            cudaMemcpyHostToDevice,
            cs));
      }

      // Replay the captured graph
      cudaError_t gerr =
          cudaGraphLaunch(handle->cuda_graph_state.graph_exec, cs);
      ET_CHECK_OR_RETURN_ERROR(
          gerr == cudaSuccess,
          Internal,
          "cudaGraphLaunch failed: %s",
          cudaGetErrorString(gerr));

      // Copy outputs back to CPU
      const bool copy_outputs =
          !should_skip_copy_for_method(handle->method_name);
      if (copy_outputs) {
        for (size_t i = 0; i < n_outputs; i++) {
          auto* cpu_out = &(args[i + n_inputs]->toTensor());
          ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMemcpyAsync(
              cpu_out->mutable_data_ptr(),
              handle->cuda_graph_state.static_output_ptrs[i],
              handle->cuda_graph_state.static_output_nbytes[i],
              cudaMemcpyDeviceToHost,
              cs));
        }
        cudaStreamSynchronize(cs);
      }

      return Error::Ok;
    }

    // ---------------------------------------------------------------
    // Normal path (also used for WARMUP and CAPTURE phases)
    // ---------------------------------------------------------------
    bool is_capture_step =
        (handle->cuda_graph_state.phase == CudaGraphPhase::Warmup &&
         handle->cuda_graph_state.warmup_remaining == 0);

    // NOTE: ExecuTorch tensors may be on CPU or GPU due to the skip-copy
    // optimization. We need to create GPU copies for CUDA kernel execution
    // using SlimTensor.
    std::vector<SlimTensor*> gpu_inputs(n_inputs);
    std::vector<SlimTensor*> gpu_outputs(n_outputs);

    // Process input tensors: convert ETensor (CPU) to SlimTensor (GPU)
    for (size_t i = 0; i < n_inputs; i++) {
      auto* cpu_tensor = &(args[i]->toTensor());

      // CAPTURE step: allocate persistent static GPU buffers
      if (is_capture_step) {
        size_t nbytes = cpu_tensor->nbytes();

        void* static_ptr = nullptr;
        cudaError_t merr = cudaMalloc(&static_ptr, nbytes);
        ET_CHECK_OR_RETURN_ERROR(
            merr == cudaSuccess,
            Internal,
            "cudaMalloc for static input %zu failed: %s",
            i,
            cudaGetErrorString(merr));

        cudaMemcpy(
            static_ptr,
            cpu_tensor->const_data_ptr(),
            nbytes,
            cudaMemcpyHostToDevice);

        handle->cuda_graph_state.static_input_ptrs.push_back(static_ptr);
        handle->cuda_graph_state.static_input_nbytes.push_back(nbytes);

        gpu_inputs[i] = make_slimtensor_from_blob_with_etensor_metadata(
            static_ptr, cpu_tensor);
        continue;
      }

      // Check if input data is already on GPU (skip-copy optimization for
      // inputs) This can happen when the caller has pre-staged data on GPU
      cudaPointerAttributes attributes{};
      const void* data_ptr = cpu_tensor->const_data_ptr();
      if (data_ptr != nullptr) {
        cudaError_t err = cudaPointerGetAttributes(&attributes, data_ptr);
        if (err == cudaSuccess && attributes.type == cudaMemoryTypeDevice) {
          // Data is already on GPU - wrap it directly without copy
          gpu_inputs[i] = make_slimtensor_from_blob_with_etensor_metadata(
              const_cast<void*>(data_ptr), cpu_tensor);

          continue;
        }
      }

      // Data is on CPU - use from_etensor to copy to GPU
      gpu_inputs[i] = new SlimTensor(
          from_etensor(*cpu_tensor, CPU_DEVICE, DEFAULT_CUDA_DEVICE));
    }

    // Process output tensors: create GPU SlimTensors for kernel output.
    // Save pre-run handles to detect orphans after run().
    std::vector<SlimTensor*> pre_run_outputs(n_outputs, nullptr);
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
      pre_run_outputs[i] = gpu_outputs[i];
    }

    bool run_called = false;

    // Scope guard: deletes any non-null gpu_outputs on exit. Normal paths
    // null entries as they take ownership, so the guard only fires on
    // early-return error paths. Also cleans up inputs if run() was never
    // called (run() steals them via internal RAII).
    executorch::backends::aoti::ScopeGuard cleanup([&]() noexcept {
      if (!run_called) {
        delete_slimtensor_vector(gpu_inputs);
      }
      for (size_t i = 0; i < gpu_outputs.size(); i++) {
        if (gpu_outputs[i]) {
          delete gpu_outputs[i];
        }
      }
    });

    // Run the AOTI container.
    // NOTE: run() steals input handles (RAII wraps them at the start of
    // run_impl) and may replace output handles with its own.
    Result<cudaStream_t> cuda_stream_ret = getCurrentCUDAStream(0);
    ET_CHECK_OK_OR_RETURN_ERROR(cuda_stream_ret.error());
    cudaStream_t cuda_stream = cuda_stream_ret.get();

    if (is_capture_step) {
      // ----- CUDA graph CAPTURE -----
      ET_LOG(
          Info,
          "CUDA graph: beginning stream capture for '%s'",
          handle->method_name.c_str());

      cudaError_t cerr =
          cudaStreamBeginCapture(cuda_stream, cudaStreamCaptureModeRelaxed);
      ET_CHECK_OR_RETURN_ERROR(
          cerr == cudaSuccess,
          Internal,
          "cudaStreamBeginCapture failed: %s",
          cudaGetErrorString(cerr));
    }

    AOTIRuntimeError error = handle->run(
        handle->container_handle,
        reinterpret_cast<Tensor**>(gpu_inputs.data()),
        n_inputs,
        reinterpret_cast<Tensor**>(gpu_outputs.data()),
        n_outputs,
        static_cast<void*>(cuda_stream),
        nullptr);
    run_called = true;

    // Delete orphaned pre-created outputs that run() replaced.
    // Must happen before the error check — if run() fails after
    // replacing some outputs, the originals would otherwise leak.
    for (size_t i = 0; i < n_outputs; i++) {
      if (pre_run_outputs[i] != gpu_outputs[i]) {
        delete pre_run_outputs[i];
      }
    }

    ET_CHECK_OR_RETURN_ERROR(
        error == Error::Ok,
        Internal,
        "AOTInductorModelContainerRun failed with error code %d",
        error);

    if (is_capture_step) {
      // End capture → instantiate graph
      cudaError_t gerr =
          cudaStreamEndCapture(cuda_stream, &handle->cuda_graph_state.graph);
      ET_CHECK_OR_RETURN_ERROR(
          gerr == cudaSuccess,
          Internal,
          "cudaStreamEndCapture failed: %s",
          cudaGetErrorString(gerr));

      gerr = cudaGraphInstantiate(
          &handle->cuda_graph_state.graph_exec,
          handle->cuda_graph_state.graph,
          cudaGraphInstantiateFlagAutoFreeOnLaunch);
      ET_CHECK_OR_RETURN_ERROR(
          gerr == cudaSuccess,
          Internal,
          "cudaGraphInstantiate failed: %s",
          cudaGetErrorString(gerr));

      // Record static output pointers (stable under graph replay)
      for (size_t i = 0; i < n_outputs; i++) {
        SlimTensor* out = gpu_outputs[i];
        handle->cuda_graph_state.static_output_ptrs.push_back(out->data_ptr());
        handle->cuda_graph_state.static_output_nbytes.push_back(out->nbytes());
      }

      handle->cuda_graph_state.phase = CudaGraphPhase::Replay;
      ET_LOG(
          Info,
          "CUDA graph: captured and instantiated for '%s'",
          handle->method_name.c_str());

      // Replay once to actually produce output (capture doesn't execute)
      gerr = cudaGraphLaunch(handle->cuda_graph_state.graph_exec, cuda_stream);
      ET_CHECK_OR_RETURN_ERROR(
          gerr == cudaSuccess,
          Internal,
          "cudaGraphLaunch (first replay) failed: %s",
          cudaGetErrorString(gerr));

      // Copy capture-step outputs to CPU
      const bool copy_outputs =
          !should_skip_copy_for_method(handle->method_name);
      if (copy_outputs) {
        for (size_t i = 0; i < n_outputs; i++) {
          auto* cpu_out = &(args[i + n_inputs]->toTensor());
          ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMemcpyAsync(
              cpu_out->mutable_data_ptr(),
              handle->cuda_graph_state.static_output_ptrs[i],
              handle->cuda_graph_state.static_output_nbytes[i],
              cudaMemcpyDeviceToHost,
              cuda_stream));
          // Don't delete — static buffers are owned by the handle
          gpu_outputs[i] = nullptr;
        }
        cudaStreamSynchronize(cuda_stream);
      } else {
        // Even when skipping copy, null out gpu_outputs to prevent
        // the ScopeGuard from deleting static output buffers.
        for (size_t i = 0; i < n_outputs; i++) {
          gpu_outputs[i] = nullptr;
        }
      }

      return Error::Ok;
    }

    // ----- Normal / WARMUP execution continues here -----

    // Decrement warmup counter if in warmup phase
    if (handle->cuda_graph_state.phase == CudaGraphPhase::Warmup &&
        handle->cuda_graph_state.warmup_remaining > 0) {
      handle->cuda_graph_state.warmup_remaining--;
      ET_LOG(
          Info,
          "CUDA graph warmup: %d steps remaining for '%s'",
          handle->cuda_graph_state.warmup_remaining,
          handle->method_name.c_str());
    }

    const bool copy_outputs = !should_skip_copy_for_method(handle->method_name);

    if (copy_outputs) {
      for (size_t i = 0; i < n_outputs; i++) {
        auto* cpu_output_tensor = &(args[i + n_inputs]->toTensor());
        ET_CHECK_OK_OR_RETURN_ERROR(
            copy_slimtensor_to_etensor_async(
                gpu_outputs[i], cpu_output_tensor, cuda_stream),
            "Failed to copy GPU output %zu back to CPU ETensor",
            i);
        delete gpu_outputs[i];
        gpu_outputs[i] = nullptr;
      }
    } else {
      // Skip-copy optimization: point ETensor directly to GPU data.
      // Lifetime management: cache GPU tensors and delete previous round's.
      {
        std::lock_guard<std::mutex> guard(cached_outputs_mutex_);
        auto& cached_outputs = cached_outputs_[handle];

        delete_slimtensor_vector(cached_outputs);

        for (size_t i = 0; i < n_outputs; i++) {
          cached_outputs.push_back(gpu_outputs[i]);
          gpu_outputs[i] = nullptr;

          auto* output_etensor = &(args[i + n_inputs]->toTensor());
          ET_CHECK_OK_OR_RETURN_ERROR(
              wrap_slimtensor_to_etensor(cached_outputs.back(), output_etensor),
              "Failed to wrap GPU output %zu into ETensor",
              i);
        }
      }
    }

    return Error::Ok;
  }

  void destroy(DelegateHandle* handle_) const override {
    if (handle_ == nullptr) {
      return;
    }
    cuda::CudaDelegateHandle* handle = (cuda::CudaDelegateHandle*)handle_;

    // Clean up cached output tensors for this handle
    {
      std::lock_guard<std::mutex> guard(cached_outputs_mutex_);
      auto it = cached_outputs_.find(handle);
      if (it != cached_outputs_.end()) {
        delete_slimtensor_vector(it->second);
        cached_outputs_.erase(it);
      }
    }

    // The CUDA stream is managed by shared_ptr in the handle.
    // It will be automatically destroyed when the last handle using it
    // is destroyed. Just reset our reference.
    handle->cuda_stream.reset();

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

  mutable std::mutex cuda_graph_method_mutex_;
  std::string cuda_graph_method_;

  // Shared CUDA stream for all methods. When set (non-null), all methods use
  // the same stream to ensure proper ordering (critical for skip-copy
  // optimization). Created when use_shared_cuda_stream option is set to true.
  // Managed via shared_ptr so it's automatically cleaned up when last handle
  // is destroyed.
  mutable std::mutex cuda_stream_mutex_;
  std::shared_ptr<cudaStream_t> shared_cuda_stream_ = nullptr;

  // Whether to enable cross-method per-FQN weight caching at init time.
  // Toggled by the kWeightSharingAcrossMethods runtime backend option. Default
  // OFF — see set_weight_sharing_across_methods() for safety constraints.
  std::atomic<bool> weight_sharing_across_methods_{false};

  // Cached output tensors for skip-copy optimization.
  // When skip-copy is enabled, output SlimTensors are cached here to keep
  // the underlying GPU memory alive while the caller processes the results.
  // Maps each CudaDelegateHandle* to its vector of cached output tensors.
  mutable std::mutex cached_outputs_mutex_;
  mutable std::
      unordered_map<cuda::CudaDelegateHandle*, std::vector<SlimTensor*>>
          cached_outputs_;

  // ---------------------------------------------------------------
  // Per-weight constant cache.
  //
  // Maintains a singleton FQN → AtenTensorHandle cache across methods.
  // When loading constants for a method, constants already in the cache
  // are reused (zero-copy via update_user_managed_constant_buffer_pairs).
  // Only constants not in the cache are loaded from the blob and added
  // to the cache. This avoids duplicate GPU allocations when multiple
  // methods (e.g., prefill/decode) share the same weights.
  //
  // ASSUMPTIONS / LIMITATIONS:
  //   * Constants with the same FQN across methods are assumed to be the
  //     SAME logical tensor (i.e. the same parameter/buffer of the same
  //     source model). We validate shape/dtype/strides/device on every
  //     reuse to catch silent mismatches (see check_cached_constant_match
  //     below). However, we cannot detect two unrelated models that
  //     happen to share an FQN.
  //   * Constants are assumed to be IMMUTABLE (parameters or read-only
  //     buffers). The AOTI shim today does not expose a mutability bit
  //     through GetConstantOriginalFQN, so we cannot detect or refuse
  //     to share mutable buffers (e.g. a per-method KV cache). If a
  //     future model exports the same FQN as a mutable buffer in
  //     multiple methods, mutations from one method WILL be visible to
  //     the other through the shared GPU memory. Callers that need
  //     per-method mutable state must currently use distinct FQNs.
  //     TODO: when AOTInductor exposes a constant-type / mutability
  //     query, refuse to share entries that are not PARAMETER or
  //     non-mutable BUFFER.
  // ---------------------------------------------------------------

  // Validates that a cached constant tensor is compatible with what the
  // new container expects for the same FQN (i.e. same dtype, dim, sizes,
  // strides, and device). Both handles point to SlimTensors in our shim
  // layer, so we can introspect them directly.
  //
  // Returns Error::Ok on a match. On mismatch, logs the offending field
  // and returns Error::Internal so callers can chain via
  // ET_CHECK_OK_OR_RETURN_ERROR and fail loudly instead of silently
  // pointing the new container at a wrong-shape buffer.
  static Error check_cached_constant_match(
      const std::string& fqn,
      AtenTensorHandle cached_handle,
      AtenTensorHandle new_handle) {
    ET_CHECK_OR_RETURN_ERROR(
        cached_handle != nullptr && new_handle != nullptr,
        Internal,
        "Constant '%s': null AtenTensorHandle (cached=%p, new=%p)",
        fqn.c_str(),
        cached_handle,
        new_handle);

    auto* cached = reinterpret_cast<SlimTensor*>(cached_handle);
    auto* fresh = reinterpret_cast<SlimTensor*>(new_handle);

    ET_CHECK_OR_RETURN_ERROR(
        cached->dtype() == fresh->dtype(),
        Internal,
        "Constant '%s': dtype mismatch (cached=%d, new=%d)",
        fqn.c_str(),
        static_cast<int>(cached->dtype()),
        static_cast<int>(fresh->dtype()));

    ET_CHECK_OR_RETURN_ERROR(
        cached->dim() == fresh->dim(),
        Internal,
        "Constant '%s': dim mismatch (cached=%zu, new=%zu)",
        fqn.c_str(),
        cached->dim(),
        fresh->dim());

    auto cached_sizes = cached->sizes();
    auto fresh_sizes = fresh->sizes();
    for (size_t i = 0; i < cached->dim(); ++i) {
      ET_CHECK_OR_RETURN_ERROR(
          cached_sizes[i] == fresh_sizes[i],
          Internal,
          "Constant '%s': size mismatch at dim %zu (cached=%lld, new=%lld)",
          fqn.c_str(),
          i,
          static_cast<long long>(cached_sizes[i]),
          static_cast<long long>(fresh_sizes[i]));
    }
    auto cached_strides = cached->strides();
    auto fresh_strides = fresh->strides();
    for (size_t i = 0; i < cached->dim(); ++i) {
      ET_CHECK_OR_RETURN_ERROR(
          cached_strides[i] == fresh_strides[i],
          Internal,
          "Constant '%s': stride mismatch at dim %zu (cached=%lld, new=%lld)",
          fqn.c_str(),
          i,
          static_cast<long long>(cached_strides[i]),
          static_cast<long long>(fresh_strides[i]));
    }
    ET_CHECK_OR_RETURN_ERROR(
        cached->device().type() == fresh->device().type() &&
            cached->device().index() == fresh->device().index(),
        Internal,
        "Constant '%s': device mismatch (cached=%d:%d, new=%d:%d)",
        fqn.c_str(),
        static_cast<int>(cached->device().type()),
        cached->device().index(),
        static_cast<int>(fresh->device().type()),
        fresh->device().index());

    return Error::Ok;
  }

  // Load constants for a method using per-weight caching.
  // Returns Error::Ok on success.
  //
  // Flow:
  //   1. Enumerate this method's constants and their FQNs.
  //   2. For each constant:
  //      - If FQN is in shared_constant_tensors_ → reuse (cache hit).
  //      - Otherwise → mark as needing loading (cache miss).
  //   3. If all constants are cached → skip blob loading entirely.
  //      Otherwise → call update_constants_from_blob to load all, then
  //      extract and cache the new constants.
  //   4. For cached constants, call update_user_managed_constant_buffer_pairs
  //      to point the container to the shared GPU tensors.
  Error load_constants_with_cache(
      cuda::CudaDelegateHandle* handle,
      const NamedDataMap* named_data_map,
      const std::string& method_name) const {
    // Check if the required APIs are available
    if (!handle->get_num_constants || !handle->get_constant_name ||
        !handle->get_constant_original_fqn || !handle->extract_constants_map ||
        !handle->update_user_managed_constant_buffer_pairs) {
      // Fall back to the legacy path
      return load_constants_legacy(handle, named_data_map, method_name);
    }

    // Step 1: Enumerate constants and partition into cached/uncached
    size_t num_constants = 0;
    handle->get_num_constants(handle->container_handle, &num_constants);
    if (num_constants == 0) {
      ET_LOG(Info, "No constants for method '%s'", method_name.c_str());
      return Error::Ok;
    }

    // Build FQN → internal_name mapping and determine cache hits/misses.
    std::unordered_map<std::string, std::string> fqn_to_name;
    std::vector<std::string> uncached_fqns;

    // Phase 1 (lock-free): enumerate constants from the container.
    for (size_t i = 0; i < num_constants; i++) {
      const char* name = nullptr;
      const char* fqn = nullptr;
      handle->get_constant_name(handle->container_handle, i, &name);
      handle->get_constant_original_fqn(handle->container_handle, i, &fqn);
      if (name && fqn && fqn[0] != '\0') {
        fqn_to_name[fqn] = name;
      }
    }

    // Phase 2 (locked): pure cache lookup against shared_constant_tensors_.
    {
      std::lock_guard<std::mutex> guard(shared_constants_mutex_);
      for (const auto& [fqn, _] : fqn_to_name) {
        if (shared_constant_tensors_.find(fqn) ==
            shared_constant_tensors_.end()) {
          uncached_fqns.push_back(fqn);
        }
      }
    }

    size_t num_cached = fqn_to_name.size() - uncached_fqns.size();
    ET_LOG(
        Info,
        "Method '%s': %zu constants, %zu cached, %zu uncached",
        method_name.c_str(),
        fqn_to_name.size(),
        num_cached,
        uncached_fqns.size());

    // Step 2: Load uncached constants from blob (if any).
    std::unordered_map<std::string, AtenTensorHandle> extracted_map;

    if (!uncached_fqns.empty()) {
      // Need to load from blob — use update_constants_from_blob for all,
      // then extract the new constants into the cache.
      std::string weights_blob_key =
          method_name.empty() ? "weights_blob" : method_name + "_weights_blob";
      auto buffer_res = named_data_map->get_data(weights_blob_key.c_str());

      ET_CHECK_OR_RETURN_ERROR(
          buffer_res.ok() && handle->update_constants_from_blob != nullptr,
          NotFound,
          "weights_blob '%s' not found or update fn is null",
          weights_blob_key.c_str());

      ET_LOG(
          Info,
          "Loading constants from blob '%s' for method '%s'",
          weights_blob_key.c_str(),
          method_name.c_str());
      const void* weights_blob = buffer_res->data();
      ET_CHECK_OK_OR_RETURN_ERROR(
          handle->update_constants_from_blob(
              handle->container_handle,
              static_cast<const uint8_t*>(weights_blob)),
          "update_constants_from_blob failed for method '%s'",
          method_name.c_str());
      cudaDeviceSynchronize();
      buffer_res->Free();

      // Extract all constants from the freshly-loaded container.
      ET_CHECK_OK_OR_RETURN_ERROR(
          handle->extract_constants_map(
              handle->container_handle,
              reinterpret_cast<AOTInductorConstantMapHandle>(&extracted_map),
              /*use_inactive=*/false),
          "Failed to extract constants from '%s'",
          method_name.c_str());

      // Validate cache hits against the freshly-extracted tensors, and
      // populate the cache with newly-loaded entries.
      {
        std::lock_guard<std::mutex> guard(shared_constants_mutex_);
        for (const auto& [fqn, _] : fqn_to_name) {
          auto extracted_it = extracted_map.find(fqn);
          if (extracted_it == extracted_map.end()) {
            // Container did not surface this FQN — skip; the user-managed
            // pair build below will simply omit it.
            continue;
          }
          auto cached_it = shared_constant_tensors_.find(fqn);
          if (cached_it == shared_constant_tensors_.end()) {
            // New constant — add to cache.
            shared_constant_tensors_[fqn] = extracted_it->second;
          } else {
            // Same FQN seen before — verify the cached tensor is still
            // compatible with what THIS method expects. On mismatch the
            // helper logs the offending field and returns an error.
            ET_CHECK_OK_OR_RETURN_ERROR(
                check_cached_constant_match(
                    fqn, cached_it->second, extracted_it->second),
                "Constant '%s' in method '%s' is incompatible with the "
                "cached version from a previous method. Refusing to share.",
                fqn.c_str(),
                method_name.c_str());
          }
        }
        ET_LOG(
            Info,
            "Cached %zu new constants from method '%s' (total cache: %zu)",
            uncached_fqns.size(),
            method_name.c_str(),
            shared_constant_tensors_.size());
      }
    } else {
      // All constants are cached — skip blob loading entirely.
      // NOTE: in this branch we cannot independently verify the cache
      // against the new container's expectations (no extract source).
      // We rely on update_user_managed_constant_buffer_pairs below,
      // which the AOTI runtime validates internally.
      ET_LOG(
          Info,
          "All %zu constants cached — skipping blob load for method '%s'",
          fqn_to_name.size(),
          method_name.c_str());
    }

    // Step 3: Point the container to cached tensors via user_managed pairs
    if (num_cached > 0 || uncached_fqns.empty()) {
      std::vector<AOTInductorConstantMapEntry> pairs;
      {
        std::lock_guard<std::mutex> guard(shared_constants_mutex_);
        for (const auto& [fqn, internal_name] : fqn_to_name) {
          auto it = shared_constant_tensors_.find(fqn);
          if (it != shared_constant_tensors_.end()) {
            pairs.push_back({internal_name.c_str(), it->second});
          }
        }
      }

      if (!pairs.empty()) {
        ET_CHECK_OK_OR_RETURN_ERROR(
            handle->update_user_managed_constant_buffer_pairs(
                handle->container_handle,
                pairs.data(),
                pairs.size(),
                /*use_inactive=*/false,
                /*validate_full_update=*/false),
            "Failed to set cached constants for method '%s'",
            method_name.c_str());
        ET_LOG(
            Info,
            "Shared %zu cached constants into method '%s'",
            pairs.size(),
            method_name.c_str());
      }
    }

    return Error::Ok;
  }

  // Legacy constant loading: load the entire blob without caching.
  // Used as fallback when constant management APIs are unavailable.
  Error load_constants_legacy(
      cuda::CudaDelegateHandle* handle,
      const NamedDataMap* named_data_map,
      const std::string& method_name) const {
    std::string weights_blob_key =
        method_name.empty() ? "weights_blob" : method_name + "_weights_blob";
    auto buffer_res = named_data_map->get_data(weights_blob_key.c_str());
    if (buffer_res.ok() && handle->update_constants_from_blob != nullptr) {
      ET_LOG(Info, "Found %s in named data map", weights_blob_key.c_str());
      const void* weights_blob = buffer_res->data();
      auto update_err = handle->update_constants_from_blob(
          handle->container_handle, static_cast<const uint8_t*>(weights_blob));
      if (update_err != Error::Ok) {
        ET_LOG(Error, "update_constants_from_blob failed");
        return update_err;
      }
      cudaDeviceSynchronize();
      buffer_res->Free();
    } else {
      ET_LOG(
          Info,
          "weights_blob '%s' not found or update fn is null",
          weights_blob_key.c_str());
    }
    return Error::Ok;
  }
  // Guards the singleton FQN → AtenTensorHandle cache below.
  //
  // The mutex guards init().
  // The CudaBackend instance is a process-wide singleton (registered
  // once via register_backend()), and shared_constant_tensors_ is a
  // shared-across-handles map. ExecuTorch hosts CAN call init() from
  // multiple threads when:
  //   * a multi-threaded application loads two Modules concurrently, or
  //   * a single Module is loaded from a thread pool.
  // Without the mutex, two concurrent init()s could race on
  // shared_constant_tensors_ (rehash during insert, double-insert with
  // different handles, etc.). The cost is a one-time lock during init,
  // which is negligible.
  mutable std::mutex shared_constants_mutex_;

  // FQN → AtenTensorHandle from the source (first) container.
  // The tensor handles are owned by the source container (which is never
  // explicitly deleted — see destroy() comment).
  mutable std::unordered_map<std::string, AtenTensorHandle>
      shared_constant_tensors_;
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
