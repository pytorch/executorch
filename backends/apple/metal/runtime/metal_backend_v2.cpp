/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// MetalBackendV2 — alternative AOTI Metal backend built on the
// MetalStream/MetalStream pipeline. Functionally equivalent to MetalBackend at
// the AOTI .so contract level (same dlopen / run / output-copy flow), but
// dispatches kernels through the simpler MetalStream surface instead of the
// ETMetalStream / MPSGraph layer.
//
// Registered as "MetalBackendV2" so it can coexist with v1 in the same binary
// in principle, but in practice you should link only one because the v2 shim
// layer (memory_v2.cpp, et_metal_v2.mm, shim_mps_v2.mm) defines the same C ABI
// symbols (aoti_torch_*, metal_*) as v1.

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

// AOTI common headers (slim, plus a local v2 delegate-handle copy that
// uses SlimTensor instead of etensor::Tensor).
#include <executorch/backends/aoti/common_shims_slim.h>
#include <executorch/backends/aoti/slim/factory/from_blob.h>
#include <executorch/backends/aoti/utils.h> // ScopeGuard

// v2 shim layer (SlimTensor-based, lives under shims/v2/)
#include <executorch/backends/apple/metal/runtime/shims/v2/aoti_tensor.h>
#include <executorch/backends/apple/metal/runtime/shims/v2/aoti_types.h>
#include <executorch/backends/apple/metal/runtime/shims/v2/delegate_handle.h>
#include <executorch/backends/apple/metal/runtime/shims/v2/runtime.h>

// MetalStream config (free fn so .cpp doesn't need MetalStream.h, which
// transitively imports Metal/Metal.h and won't compile in non-ObjC++ TUs).

namespace executorch::backends::metal {

#define LOAD_SYMBOL(handle, member, name, so_handle)                        \
  do {                                                                      \
    handle->member = reinterpret_cast<name##Func>(dlsym(so_handle, #name)); \
    ET_CHECK_OR_RETURN_ERROR(                                               \
        handle->member != nullptr, AccessFailed, "Failed to load " #name);  \
  } while (0)

using namespace std;
using namespace aoti;
namespace slim = executorch::backends::aoti::slim;

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
// ETensor: the Method-allocated CPU input/output tensor type. Distinct from
// the v2 backend's `Tensor` (= SlimTensor, from types_v2.h via memory_v2.h).
using ETensor = executorch::runtime::etensor::Tensor;

namespace {
// Build a zero-copy SlimTensor view over a CPU-resident ETensor's storage.
// Used at the I/O boundary so we can hand a SlimTensor handle to the v2
// AOTI shims (aoti_torch_copy_) without copying data.
slim::SlimTensor slim_view_of_etensor(const ETensor& e) {
  std::vector<int64_t> sizes(e.sizes().begin(), e.sizes().end());
  std::vector<int64_t> strides(e.strides().begin(), e.strides().end());
  void* data = const_cast<void*>(static_cast<const void*>(e.const_data_ptr()));
  return slim::from_blob(
      data,
      slim::makeArrayRef(sizes),
      slim::makeArrayRef(strides),
      static_cast<slim::c10::ScalarType>(static_cast<int>(e.scalar_type())));
}
} // namespace

class ET_EXPERIMENTAL MetalBackendV2 final
    : public ::executorch::runtime::BackendInterface {
 private:
  Error load_function_pointers_into_handle(
      void* so_handle,
      AOTIDelegateHandle* handle) const {
    LOAD_SYMBOL(
        handle,
        create_with_device,
        AOTInductorModelContainerCreateWithDevice,
        so_handle);
    LOAD_SYMBOL(
        handle, delete_container, AOTInductorModelContainerDelete, so_handle);
    LOAD_SYMBOL(
        handle,
        get_num_inputs,
        AOTInductorModelContainerGetNumInputs,
        so_handle);
    LOAD_SYMBOL(
        handle,
        get_num_outputs,
        AOTInductorModelContainerGetNumOutputs,
        so_handle);
    LOAD_SYMBOL(handle, run, AOTInductorModelContainerRun, so_handle);
    LOAD_SYMBOL(
        handle,
        update_constants_from_blob,
        AOTInductorModelUpdateConstantsFromBlob,
        so_handle);
    return Error::Ok;
  }

 public:
  MetalBackendV2() {
    ET_LOG(Debug, "MetalBackendV2 ctor");
  }

  bool is_available() const override {
    return 1;
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
    ET_LOG(Info, "MetalBackendV2::init - Starting initialization");

    std::string method_name;
    for (const CompileSpec& spec : compile_specs) {
      if (std::strcmp(spec.key, "method_name") == 0) {
        method_name.assign(
            static_cast<const char*>(spec.value.buffer), spec.value.nbytes);
        break;
      }
    }

    std::string so_blob_key =
        method_name.empty() ? "so_blob" : method_name + "_so_blob";

    const NamedDataMap* named_data_map = context.get_named_data_map();
    ET_CHECK_OR_RETURN_ERROR(
        named_data_map != nullptr,
        Internal,
        "MetalBackendV2 requires a NamedDataMap for weight loading");

    // Prefetch the weights blob — same trick as v1.
    std::string weights_blob_key =
        method_name.empty() ? "weights_blob" : method_name + "_weights_blob";
    {
      auto prefetch_buf = named_data_map->get_data(weights_blob_key.c_str());
      if (prefetch_buf.ok() && prefetch_buf->data() != nullptr) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(prefetch_buf->data());
        size_t page_size = getpagesize();
        uintptr_t aligned_addr = addr & ~(page_size - 1);
        size_t aligned_size = prefetch_buf->size() + (addr - aligned_addr);
        int ret = madvise(
            reinterpret_cast<void*>(aligned_addr), aligned_size, MADV_WILLNEED);
        if (ret != 0) {
          ET_LOG(
              Info,
              "MetalBackendV2::init - madvise failed for %s: %s",
              weights_blob_key.c_str(),
              strerror(errno));
        }
      }
    }

    auto aoti_metal_buffer = named_data_map->get_data(so_blob_key.c_str());
    ET_CHECK_OR_RETURN_ERROR(
        aoti_metal_buffer.ok(),
        Internal,
        "Failed to get data for key %s: 0x%x",
        so_blob_key.c_str(),
        static_cast<uint32_t>(aoti_metal_buffer.error()));

    if (aoti_metal_buffer->data() == nullptr) {
      ET_LOG(Error, "MetalBackendV2::init - Buffer data is null");
      return Error::InvalidArgument;
    }

    filesystem::path temp_dir = filesystem::temp_directory_path();
    filesystem::path so_path =
        temp_dir / (so_blob_key + to_string(getpid()) + "_v2.so");

    ET_LOG(
        Info, "MetalBackendV2::init - Creating temp file: %s", so_path.c_str());
    ofstream outfile(so_path.c_str(), ios::binary);

    outfile.write(
        static_cast<const char*>(aoti_metal_buffer->data()),
        aoti_metal_buffer->size());

    ET_CHECK_OR_RETURN_ERROR(
        outfile, AccessFailed, "Failed to write to file %s", so_path.c_str());

    outfile.close();
    aoti_metal_buffer->Free();

    void* so_handle = dlopen(so_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    ET_CHECK_OR_RETURN_ERROR(
        so_handle != nullptr,
        AccessFailed,
        "Failed to load shared library: %s",
        dlerror());

    ET_LOG(
        Info,
        "MetalBackendV2::init - Loaded shared library: %s",
        so_path.c_str());

    processed->Free();

    AOTIDelegateHandle* handle = new AOTIDelegateHandle();
    handle->so_handle = so_handle;
    handle->so_path = so_path.string();

    ET_CHECK_OK_OR_RETURN_ERROR(
        load_function_pointers_into_handle(so_handle, handle));

    AOTInductorModelContainerHandle container_handle = nullptr;
    ET_LOG(
        Info,
        "MetalBackendV2::init - About to create AOTI container with device='mps'");

    ET_CHECK_OK_OR_RETURN_ERROR(
        handle->create_with_device(&container_handle, 1, "mps", nullptr));

    handle->container_handle = container_handle;

    auto buffer_res = named_data_map->get_data(weights_blob_key.c_str());
    if (buffer_res.ok() && handle->update_constants_from_blob != nullptr) {
      const void* weights_blob = buffer_res->data();
      ET_CHECK_OK_OR_RETURN_ERROR(handle->update_constants_from_blob(
          handle->container_handle, static_cast<const uint8_t*>(weights_blob)));
      buffer_res->Free();
    }

    ET_LOG(Info, "MetalBackendV2::init - Initialization completed");
    return (DelegateHandle*)handle;
  }

  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* handle_,
      Span<EValue*> args) const override {
    ET_LOG(Debug, "MetalBackendV2 execute");

    // Apply ET_METAL_FLUSH_INTERVAL to the current thread's stream.
    // Parse the env var exactly once, then call setFlushInterval() on every
    // execute() — it's idempotent and cheap (just an int store), and ensures
    // every thread's stream picks up the override on its first execute().
    static const int kEnvFlushInterval = []() -> int {
      const char* env = std::getenv("ET_METAL_FLUSH_INTERVAL");
      if (!env)
        return -1;
      try {
        int v = std::stoi(env);
        return v >= 0 ? v : -1;
      } catch (const std::exception&) {
        ET_LOG(Error, "Invalid ET_METAL_FLUSH_INTERVAL value: '%s'", env);
        return -1;
      }
    }();
    if (kEnvFlushInterval >= 0) {
      metal::metal_set_flush_interval(kEnvFlushInterval);
    }

    AOTIDelegateHandle* handle = (AOTIDelegateHandle*)handle_;

    size_t n_inputs;
    handle->get_num_inputs(handle->container_handle, &n_inputs);

    size_t n_outputs;
    handle->get_num_outputs(handle->container_handle, &n_outputs);

    ET_CHECK_OR_RETURN_ERROR(
        n_inputs + n_outputs == args.size(),
        InvalidArgument,
        "input %zd + output %zd != args.size() %zd",
        n_inputs,
        n_outputs,
        args.size());

    int32_t mps_device_type = aoti_torch_device_type_mps();

    std::vector<AOTITensorHandle> gpu_inputs(n_inputs, nullptr);
    std::vector<AOTITensorHandle> gpu_outputs(n_outputs, nullptr);
    std::vector<AOTITensorHandle> pre_run_outputs(n_outputs, nullptr);

    bool run_called = false;

    executorch::backends::aoti::ScopeGuard cleanup([&]() noexcept {
      if (!run_called) {
        for (size_t i = 0; i < gpu_inputs.size(); i++) {
          if (gpu_inputs[i]) {
            aoti_torch_delete_tensor_object(gpu_inputs[i]);
          }
        }
      }
      for (size_t i = 0; i < gpu_outputs.size(); i++) {
        if (pre_run_outputs[i] && pre_run_outputs[i] != gpu_outputs[i]) {
          aoti_torch_delete_tensor_object(pre_run_outputs[i]);
        }
        if (gpu_outputs[i]) {
          aoti_torch_delete_tensor_object(gpu_outputs[i]);
        }
      }
    });

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

      // Wrap the CPU ETensor in a temporary SlimTensor view so we can hand
      // it to the SlimTensor-typed aoti_torch_copy_ shim (zero copy; the
      // view dies when this scope ends).
      auto cpu_view = slim_view_of_etensor(*cpu_tensor);
      ET_CHECK_OK_OR_RETURN_ERROR(
          aoti_torch_copy_(gpu_inputs[i], &cpu_view, 0),
          "Failed to copy input %d from CPU to GPU",
          i);
    }

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

    try {
      synchronize_metal_stream();
    } catch (const std::exception& e) {
      ET_LOG(Error, "Failed to synchronize Metal stream: %s", e.what());
      return Error::Internal;
    } catch (...) {
      ET_LOG(Error, "Failed to synchronize Metal stream: unknown exception");
      return Error::Internal;
    }

    for (size_t i = 0; i < n_outputs; i++) {
      auto cpu_output_tensor = &(args[i + n_inputs]->toTensor());
      // gpu_outputs[i] is a SlimTensor*; its sizes() is IntArrayRef (int64),
      // but resize_tensor expects ETensor's int32 SizesType. Convert.
      auto gpu_sizes = gpu_outputs[i]->sizes();
      std::vector<executorch::aten::SizesType> gpu_sizes_et(
          gpu_sizes.begin(), gpu_sizes.end());
      ET_CHECK_OK_OR_RETURN_ERROR(
          resize_tensor(
              *cpu_output_tensor,
              ArrayRef<executorch::aten::SizesType>(
                  gpu_sizes_et.data(), gpu_sizes_et.size())),
          "Error resizing tensor at output index %d",
          i);
      // Wrap the CPU ETensor as a SlimTensor view for the copy call.
      auto cpu_view = slim_view_of_etensor(*cpu_output_tensor);
      ET_CHECK_OK_OR_RETURN_ERROR(
          aoti_torch_copy_(&cpu_view, gpu_outputs[i], 0),
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

    // Same workaround as v1: skip explicit container deletion to avoid
    // shared-state segfaults across multiple .so files.
    // AOTInductorModelContainerDelete(handle->container_handle);

    if (handle->so_handle != nullptr) {
      dlclose(handle->so_handle);
    }

    if (!handle->so_path.empty()) {
      std::error_code remove_error;
      std::filesystem::remove(handle->so_path, remove_error);
    }

    delete handle;
    cleanup_memory();
    ET_LOG(Debug, "MetalBackendV2 handle %p destroy", handle_);
  }
};

} // namespace executorch::backends::metal

namespace executorch::backends {
namespace {
auto cls_v2 = metal::MetalBackendV2();
// Register under the same name as v1 ("MetalBackend") so existing partitioner
// output / .pte files transparently route through v2. v1 and v2 cannot be
// linked into the same binary (their C-ABI shim symbols collide), so reusing
// the name is safe — at most one is registered per process.
executorch::runtime::Backend backend_v2{"MetalBackend", &cls_v2};
static executorch::runtime::Error success_with_compiler_v2 =
    register_backend(backend_v2);
} // namespace
} // namespace executorch::backends
