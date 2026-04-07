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
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include <filesystem>
#include <fstream>
#include <mutex>
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

#ifdef EXECUTORCH_METAL_COLLECT_STATS

// Per-method timing statistics
struct MethodStats {
  double total_ms = 0.0;
  int64_t call_count = 0;
};

// Singleton struct containing all timing statistics and mutex
struct StatsData {
  std::mutex mutex;
  double execute_total_ms = 0.0;
  int64_t execute_call_count = 0;
  double init_total_ms = 0.0;
  int64_t init_call_count = 0;
  std::unordered_map<std::string, MethodStats> method_stats;
  std::unordered_map<std::string, MethodStats> init_method_stats;
};

// Thread-safe singleton accessor using C++11 magic statics
static StatsData& get_stats_data() {
  static StatsData instance;
  return instance;
}

// Accessor functions for execute timing statistics
double get_metal_backend_execute_total_ms() {
  auto& stats = get_stats_data();
  std::lock_guard<std::mutex> lock(stats.mutex);
  return stats.execute_total_ms;
}

int64_t get_metal_backend_execute_call_count() {
  auto& stats = get_stats_data();
  std::lock_guard<std::mutex> lock(stats.mutex);
  return stats.execute_call_count;
}

// Accessor functions for init timing statistics
double get_metal_backend_init_total_ms() {
  auto& stats = get_stats_data();
  std::lock_guard<std::mutex> lock(stats.mutex);
  return stats.init_total_ms;
}

int64_t get_metal_backend_init_call_count() {
  auto& stats = get_stats_data();
  std::lock_guard<std::mutex> lock(stats.mutex);
  return stats.init_call_count;
}

void reset_metal_backend_stats() {
  auto& stats = get_stats_data();
  std::lock_guard<std::mutex> lock(stats.mutex);
  stats.execute_total_ms = 0.0;
  stats.execute_call_count = 0;
  stats.init_total_ms = 0.0;
  stats.init_call_count = 0;
  stats.method_stats.clear();
  stats.init_method_stats.clear();
}

std::unordered_map<std::string, std::pair<double, int64_t>>
get_metal_backend_per_method_stats() {
  auto& stats = get_stats_data();
  std::lock_guard<std::mutex> lock(stats.mutex);
  std::unordered_map<std::string, std::pair<double, int64_t>> result;
  for (const auto& entry : stats.method_stats) {
    result[entry.first] = {entry.second.total_ms, entry.second.call_count};
  }
  return result;
}

std::unordered_map<std::string, std::pair<double, int64_t>>
get_metal_backend_init_per_method_stats() {
  auto& stats = get_stats_data();
  std::lock_guard<std::mutex> lock(stats.mutex);
  std::unordered_map<std::string, std::pair<double, int64_t>> result;
  for (const auto& entry : stats.init_method_stats) {
    result[entry.first] = {entry.second.total_ms, entry.second.call_count};
  }
  return result;
}

#endif // EXECUTORCH_METAL_COLLECT_STATS

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
#ifdef EXECUTORCH_METAL_COLLECT_STATS
    auto init_start = std::chrono::high_resolution_clock::now();
#endif
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
    ET_CHECK_OR_RETURN_ERROR(
        named_data_map != nullptr,
        Internal,
        "MetalBackend requires a NamedDataMap for weight loading");

    // Prefetch the weights blob — trigger async readahead so pages are
    // resident by the time update_constants_from_blob memcpy's them.
    // This overlaps disk I/O with the .so write + dlopen (~200ms).
    std::string weights_blob_key =
        method_name.empty() ? "weights_blob" : method_name + "_weights_blob";
    {
      auto prefetch_buf = named_data_map->get_data(weights_blob_key.c_str());
      if (prefetch_buf.ok() && prefetch_buf->data() != nullptr) {
        // Align address down to page boundary (madvise requires it).
        uintptr_t addr = reinterpret_cast<uintptr_t>(prefetch_buf->data());
        size_t page_size = getpagesize();
        uintptr_t aligned_addr = addr & ~(page_size - 1);
        size_t aligned_size = prefetch_buf->size() + (addr - aligned_addr);
        int ret = madvise(
            reinterpret_cast<void*>(aligned_addr), aligned_size, MADV_WILLNEED);
        if (ret != 0) {
          ET_LOG(
              Info,
              "MetalBackend::init - madvise(MADV_WILLNEED) failed for %s: %s",
              weights_blob_key.c_str(),
              strerror(errno));
        } else {
          ET_LOG(
              Info,
              "MetalBackend::init - Prefetching %s (%.1f MB)",
              weights_blob_key.c_str(),
              prefetch_buf->size() / (1024.0 * 1024.0));
        }
      }
    }

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

    ET_LOG(
        Info,
        "MetalBackend::init - Loaded shared library: %s",
        so_path.c_str());

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

    // Look into named data map for constant data (key computed above for
    // prefetch)
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

#ifdef EXECUTORCH_METAL_COLLECT_STATS
    // Accumulate init timing statistics
    auto init_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(init_end - init_start)
            .count();

    {
      auto& stats_data = get_stats_data();
      std::lock_guard<std::mutex> lock(stats_data.mutex);
      stats_data.init_total_ms += elapsed_ms;
      stats_data.init_call_count++;

      // Track per-method init timing
      if (!method_name.empty()) {
        auto& method_stats = stats_data.init_method_stats[method_name];
        method_stats.total_ms += elapsed_ms;
        method_stats.call_count++;
      }
    }
#endif

    return (DelegateHandle*)handle; // Return the handle post-processing
  }

  // Once per execution
  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* handle_,
      Span<EValue*> args) const override {
#ifdef EXECUTORCH_METAL_COLLECT_STATS
    auto execute_start = std::chrono::high_resolution_clock::now();
#endif
    ET_LOG(Debug, "MetalBackend execute");

    // Allow overriding the default flush interval (set in ETMetalStream
    // constructor)
    static std::once_flag flush_interval_flag;
    std::call_once(flush_interval_flag, [] {
      const char* env = std::getenv("ET_METAL_FLUSH_INTERVAL");
      if (env) {
        try {
          int val = std::stoi(env);
          if (val >= 0) {
            getCurrentMetalStream()->setFlushInterval(val);
          }
        } catch (const std::exception&) {
          ET_LOG(Error, "Invalid ET_METAL_FLUSH_INTERVAL value: '%s'", env);
        }
      }
    });

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

    // NOTE: ExecuTorch tensors are always on CPU/host memory.
    // We create GPU copies for Metal kernel execution.
    std::vector<AOTITensorHandle> gpu_inputs(n_inputs, nullptr);
    std::vector<AOTITensorHandle> gpu_outputs(n_outputs, nullptr);
    // Saved pre-run output handles so we can detect (and clean up)
    // outputs that run() replaces with its own tensors.
    std::vector<AOTITensorHandle> pre_run_outputs(n_outputs, nullptr);

    // Track whether run() has been called. Before run(), we own the
    // inputs and must clean them up on error. After run(), inputs are
    // stolen (RAII inside run_impl deletes them).
    bool run_called = false;

    // Scope guard: ensures all GPU tensors are cleaned up on any exit path.
    executorch::backends::aoti::ScopeGuard cleanup([&]() noexcept {
      // Clean up inputs only if run() was never called (it steals them).
      if (!run_called) {
        for (size_t i = 0; i < gpu_inputs.size(); i++) {
          if (gpu_inputs[i]) {
            aoti_torch_delete_tensor_object(gpu_inputs[i]);
          }
        }
      }
      // Clean up outputs: delete orphaned pre-created tensors that run()
      // replaced, and delete the current output handles.
      for (size_t i = 0; i < gpu_outputs.size(); i++) {
        if (pre_run_outputs[i] && pre_run_outputs[i] != gpu_outputs[i]) {
          aoti_torch_delete_tensor_object(pre_run_outputs[i]);
        }
        if (gpu_outputs[i]) {
          aoti_torch_delete_tensor_object(gpu_outputs[i]);
        }
      }
    });

    // Create GPU input tensors and copy CPU data to them.
    for (size_t i = 0; i < n_inputs; i++) {
      auto cpu_tensor = &(args[i]->toTensor());
      auto sizes = cpu_tensor->sizes();
      std::vector<int64_t> sizes_vec(sizes.begin(), sizes.end());

      ET_CHECK_OK_OR_RETURN_ERROR(
          aoti_torch_empty_strided(
              sizes_vec.size(),
              sizes_vec.data(),
              nullptr,
              static_cast<int32_t>(cpu_tensor->scalar_type()),
              mps_device_type,
              0,
              &gpu_inputs[i]),
          "Failed to create GPU tensor for input %d",
          i);

      ET_CHECK_OK_OR_RETURN_ERROR(
          aoti_torch_copy_(gpu_inputs[i], cpu_tensor, 0),
          "Failed to copy input %d from CPU to GPU",
          i);
    }

    // Create GPU output tensors. run() may replace these with its own
    // handles — pre_run_outputs lets us detect and clean up orphans.
    for (size_t i = 0; i < n_outputs; i++) {
      auto cpu_output_tensor = &(args[i + n_inputs]->toTensor());
      auto sizes = cpu_output_tensor->sizes();
      std::vector<int64_t> sizes_vec(sizes.begin(), sizes.end());

      ET_CHECK_OK_OR_RETURN_ERROR(
          aoti_torch_empty_strided(
              sizes_vec.size(),
              sizes_vec.data(),
              nullptr,
              static_cast<int32_t>(cpu_output_tensor->scalar_type()),
              mps_device_type,
              0,
              &gpu_outputs[i]),
          "Failed to create GPU tensor for output %d",
          i);

      pre_run_outputs[i] = gpu_outputs[i];
    }

    // Run AOTI container. Per the AOTI contract:
    //   - input handles are "stolen" (run() takes ownership via RAII)
    //   - output handles are written by run() (may replace pre-created ones)
    // NOTE: We assume run() steals all inputs upfront (RAII wraps them at
    // the start of run_impl). If run() fails partway, un-stolen inputs
    // would leak — but the AOTI contract guarantees all-or-nothing
    // ownership transfer.
    AOTIRuntimeError error = handle->run(
        handle->container_handle,
        gpu_inputs.data(),
        n_inputs,
        gpu_outputs.data(),
        n_outputs,
        nullptr,
        nullptr);
    run_called = true;

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

    // Copy GPU output results back to CPU output tensors
    for (size_t i = 0; i < n_outputs; i++) {
      auto cpu_output_tensor = &(args[i + n_inputs]->toTensor());
      ET_CHECK_OK_OR_RETURN_ERROR(
          resize_tensor(*cpu_output_tensor, gpu_outputs[i]->sizes()),
          "Error resizing tensor at output index %d",
          i);
      ET_CHECK_OK_OR_RETURN_ERROR(
          aoti_torch_copy_(cpu_output_tensor, gpu_outputs[i], 0),
          "Failed to copy GPU output %d back to CPU",
          i);
    }

    // ScopeGuard destructor cleans up all output tensors (both
    // orphaned pre-created ones and run()'s replacements).

    ET_LOG(Debug, "MetalBackend execution completed successfully");

#ifdef EXECUTORCH_METAL_COLLECT_STATS
    // Accumulate timing statistics
    auto execute_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(execute_end - execute_start)
            .count();

    {
      auto& stats_data = get_stats_data();
      std::lock_guard<std::mutex> lock(stats_data.mutex);
      stats_data.execute_total_ms += elapsed_ms;
      stats_data.execute_call_count++;

      // Track per-method timing
      const char* method_name = context.get_method_name();
      if (method_name != nullptr) {
        auto& method_stats = stats_data.method_stats[method_name];
        method_stats.total_ms += elapsed_ms;
        method_stats.call_count++;
      }
    }
#endif

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
