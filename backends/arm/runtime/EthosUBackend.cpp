/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2023-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Common Arm backend for Ethos-U. The driver-specific platform implementations
 * live in EthosUBackend_CoreDriver.cpp and EthosUBackend_LinuxDriver.cpp.
 */

#include <cstdint>
#include <cstring>
#include <new>

#include <executorch/backends/arm/runtime/EthosUBackend_Internal.h>
#include <executorch/backends/arm/runtime/EthosUBackend_Pmu.h>
#include <executorch/backends/arm/runtime/VelaBinStream.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

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
using executorch::runtime::Result;
using executorch::runtime::Span;

namespace executorch {
namespace backends {
namespace arm {

// cppcheck-suppress unusedFunction
bool __attribute__((weak)) ethosu_execution_metrics_enabled() noexcept {
  return false;
}

extern "C" {
/** Optional hook called immediately before common delegate execution. */
void __attribute__((weak)) EthosUBackend_execute_begin() {}
/** Optional hook called whenever common delegate execution exits. */
void __attribute__((weak)) EthosUBackend_execute_end() {}
#if defined(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
void __attribute__((weak)) EthosUBackend_delegate_begin(const void*) {}
void __attribute__((weak)) EthosUBackend_delegate_end() {}
#endif
#if defined(ET_ARM_ETHOSU_PROFILE_IO_COPIES)
void __attribute__((weak)) EthosUBackend_input_memcpy(size_t) {}
void __attribute__((weak)) EthosUBackend_output_memcpy(size_t) {}
#endif
__attribute__((weak)) unsigned char* ethosu_fast_scratch = nullptr;
__attribute__((weak)) size_t ethosu_fast_scratch_size = 0;
}

class EthosUBackendExecuteCallbacks {
 public:
#if defined(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
  explicit EthosUBackendExecuteCallbacks(const void* handle) {
    EthosUBackend_execute_begin();
    EthosUBackend_delegate_begin(handle);
  }
#else
  EthosUBackendExecuteCallbacks() {
    EthosUBackend_execute_begin();
  }
#endif
  ~EthosUBackendExecuteCallbacks() {
#if defined(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
    EthosUBackend_delegate_end();
#endif
    EthosUBackend_execute_end();
  }
};

namespace {

/**
 * Returns supported planned scratch, allocates temporary addressable scratch
 * when required, or returns empty scratch for driver-managed platforms.
 */
Result<Span<uint8_t>> get_ethosu_scratch(
    BackendExecutionContext& context,
    const VelaHandles& handles,
    bool needs_scratch_allocation) {
  auto planned_scratch = context.scratch_buffers();
  // Use planned scratch when the PTE provides it and the platform needs it.
  if (!planned_scratch.empty() && needs_scratch_allocation) {
    if (planned_scratch.size() != 1) {
      ET_LOG(Error, "Ethos-U requires exactly one planned scratch buffer");
      return Error::InvalidProgram;
    }
    Span<uint8_t> scratch = planned_scratch[0];
    if (scratch.size() < handles.scratch_data_size ||
        (scratch.size() > 0 && scratch.data() == nullptr) ||
        reinterpret_cast<uintptr_t>(scratch.data()) %
                ETHOSU_SCRATCH_ALIGNMENT !=
            0) {
      ET_LOG(
          Error,
          "Ethos-U planned scratch is invalid: %zu bytes, expected at least %zu bytes aligned to 16 bytes",
          scratch.size(),
          handles.scratch_data_size);
      return Error::InvalidProgram;
    }
    return scratch;
  }

  // Driver-managed or empty scratch requires no allocation here.
  if (!needs_scratch_allocation || handles.scratch_data_size == 0) {
    return Span<uint8_t>();
  }

  // Use temporary scratch when the PTE does not provide planned scratch.
  MemoryAllocator* temp_allocator = context.get_temp_allocator();
  if (temp_allocator == nullptr) {
    return Error::MemoryAllocationFailed;
  }
  void* scratch_data = temp_allocator->allocate(
      handles.scratch_data_size, ETHOSU_SCRATCH_ALIGNMENT);
  if (scratch_data == nullptr) {
    ET_LOG(
        Error,
        "Failed to allocate scratch buffer of %zu bytes from temp_allocator",
        handles.scratch_data_size);
    return Error::MemoryAllocationFailed;
  }

  return Span<uint8_t>(
      static_cast<uint8_t*>(scratch_data), handles.scratch_data_size);
}

} // namespace

class EthosUBackend final : public ::executorch::runtime::BackendInterface {
 public:
  EthosUBackend() {}

  ~EthosUBackend() = default;

  virtual bool is_available() const override {
    // TODO: revise to use a register check/init function
    return 1;
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
#if defined(ET_EVENT_TRACER_ENABLED)
    EventTracer* event_tracer = context.event_tracer();
    EventTracerEntry event_tracer_local_scope;
#endif

    EXECUTORCH_PROF_START(
        event_tracer,
        event_tracer_local_scope,
        "+EthosUBackend::init()processed_data");
    const char* data = static_cast<const char*>(processed->data());
    EXECUTORCH_PROF_END(event_tracer, event_tracer_local_scope);

    ET_LOG(Info, "data:%p", data);
    size_t size = processed->size();

    // Verify format of vela_bin
    if (vela_bin_validate(data, size) == false) {
      ET_LOG(Error, "Malformed vela_bin_stream found");
      return Error::InvalidProgram;
    }

    EXECUTORCH_PROF_START(
        event_tracer,
        event_tracer_local_scope,
        "+EthosUBackend::init()vela_bin_read()");
    VelaHandles handles{};
    const Error read_status =
        vela_bin_read(data, size, context.get_named_data_map(), &handles);
    EXECUTORCH_PROF_END(event_tracer, event_tracer_local_scope);
    if (read_status != Error::Ok) {
      return read_status;
    }

    MemoryAllocator* allocator = context.get_runtime_allocator();
    if (allocator == nullptr) {
      return Error::MemoryAllocationFailed;
    }
    ExecutionHandle* handle = allocator->allocateInstance<ExecutionHandle>();
    if (handle == nullptr) {
      return Error::MemoryAllocationFailed;
    }
    new (handle) ExecutionHandle();
    handle->handles = handles;
    const Error platform_status =
        platform_init(compile_specs, allocator, handle);
    if (platform_status != Error::Ok) {
      handle->~ExecutionHandle();
      return platform_status;
    }

#ifdef ETHOSU_PERSISTENT_REGION
    // Delegate-owned streaming state lives in its own region (Vela region 5,
    // base address 5), separate from scratch (base 1) and the read-only weights
    // (base 0). Allocate it from the runtime allocator so it lives for the
    // model lifetime, and install its initial contents once here. Assumes a
    // single model owns the persistent region. The persistent_size and
    // persistent_init blocks require Vela 5.3 or later
    // (--separate-persistent-region).
    //
    // An init image must cover the region exactly. This also rejects an image
    // with no persistent_size block, which would otherwise be dropped silently.
    if (handle->handles.persistent_init_data != nullptr &&
        handle->handles.persistent_init_size !=
            handle->handles.persistent_data_size) {
      ET_LOG(
          Error,
          "Persistent init image is %u bytes but the region is %u bytes",
          static_cast<unsigned>(handle->handles.persistent_init_size),
          static_cast<unsigned>(handle->handles.persistent_data_size));
      handle->~ExecutionHandle();
      return Error::InvalidProgram;
    }
    if (handle->handles.persistent_data_size > 0) {
      handle->persistent_region = static_cast<char*>(
          allocator->allocate(handle->handles.persistent_data_size, 16UL));
      if (handle->persistent_region == nullptr) {
        ET_LOG(
            Error,
            "Failed to allocate %u bytes of persistent state region",
            static_cast<unsigned>(handle->handles.persistent_data_size));
        handle->~ExecutionHandle();
        return Error::MemoryAllocationFailed;
      }
      // Install the initial state. Vela supplies an image only when the state
      // does not start from the all-zero bit pattern, which is the case
      // whenever a state has a non-zero initial value or a non-zero quantized
      // zero point. Otherwise zeroing is equivalent and the blob stays smaller.
      if (handle->handles.persistent_init_data != nullptr) {
        std::memcpy(
            handle->persistent_region,
            handle->handles.persistent_init_data,
            handle->handles.persistent_init_size);
      } else {
        std::memset(
            handle->persistent_region, 0, handle->handles.persistent_data_size);
      }
      ET_LOG(
          Info,
          "Persistent state region: %u bytes at %p, %s",
          static_cast<unsigned>(handle->handles.persistent_data_size),
          handle->persistent_region,
          handle->handles.persistent_init_data != nullptr ? "initialised"
                                                          : "zeroed");
    }
#endif

    // Return the same buffer we were passed - this data will be
    // executed directly
    return handle;
  }

  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* input_handle,
      Span<EValue*> args) const override {
#if defined(ET_EVENT_TRACER_ENABLED)
    EventTracer* event_tracer = context.event_tracer();
#endif

    EXECUTORCH_PROF_SCOPE(event_tracer, "EthosUBackend::execute()");

    // CollectArm_CPU_Cycles is just used to save the numbers of CPU cycles
    // used, If etdump is used the EXECUTORCH_PROF_SCOPE() above will do the
    // same. If not, this is a cheap way of getting some stats and the
    // CollectArm_CPU_Cycles object can safely be removed in production code.
    //
    // The EthosUBackendExecuteCallbacks class uses the C++
    // constructor/destructor to make sure that EthosUBackend_execute_begin()
    // and EthosUBackend_execute_end() is called while CollectArm_CPU_Cycles is
    // in scope. e.g. We meassure from now until we exit this metod (in any way
    // we might do it).
#if defined(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
    EthosUBackendExecuteCallbacks CollectArm_CPU_Cycles(input_handle);
#else
    EthosUBackendExecuteCallbacks CollectArm_CPU_Cycles;
#endif

    ExecutionHandle* execution_handle =
        static_cast<ExecutionHandle*>(input_handle);
    const VelaHandles& handles = execution_handle->handles;

    auto arguments = parse_ethosu_arguments(handles, args);
    if (!arguments.ok()) {
      return arguments.error();
    }

    const bool needs_scratch = needs_scratch_allocation();
    Result<Span<uint8_t>> scratch_result =
        get_ethosu_scratch(context, handles, needs_scratch);
    if (!scratch_result.ok()) {
      return scratch_result.error();
    }
    Span<uint8_t> ethosu_scratch = scratch_result.get();

    auto io_plan =
        make_io_plan(handles, *arguments, ethosu_scratch, needs_scratch);
    if (!io_plan.ok()) {
      return io_plan.error();
    }

    return platform_execute(context, execution_handle, *io_plan);
  }

  void destroy(DelegateHandle* handle) const override {
    if (handle == nullptr) {
      return;
    }

    // Explicitly destroy platform-specific state before releasing the
    // execution handle to avoid leaking resources such as std::string.
    auto* exec_handle = reinterpret_cast<ExecutionHandle*>(handle);

    if (exec_handle->platform_state != nullptr) {
      platform_destroy(exec_handle->platform_state);
    }

    std::destroy_at(exec_handle);
  }

 private:
  // No platform-specific members.
};

namespace {
auto EthosUBackend_backend = EthosUBackend();
Backend EthosUBackend_id{"EthosUBackend", &EthosUBackend_backend};
static executorch::runtime::Error EthosUBackend_registered =
    register_backend(EthosUBackend_id);

} // namespace

} // namespace arm
} // namespace backends
} // namespace executorch
