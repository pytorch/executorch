/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Arm backend for Ethos-U bare-metal (no-OS) and RTOS targets. This relies on
 * the ethos-u-core-driver for hardware interaction.
 */

#include <cstdint>
#include <memory>

#include <ethosu_driver.h>

#include <executorch/backends/arm/runtime/EthosUBackend_Internal.h>
#include <executorch/backends/arm/runtime/EthosUBackend_Pmu.h>
#include <executorch/backends/arm/runtime/EthosUIOPlan.h>
#include <executorch/runtime/core/error.h>

using executorch::runtime::BackendExecutionContext;
using executorch::runtime::Error;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Result;

// Compatibility hooks for multi-device driver / non-multi-device driver code
// When multi-device driver code is available, these declarations are overridden
extern "C" __attribute__((weak)) int ethosu_get_product_config_from_cop_data(
    const void*,
    const int,
    uint32_t* product_out,
    uint32_t* log2_macs_out) {
  *product_out = 0;
  *log2_macs_out = 0;
  return 0;
}

extern "C" __attribute__((weak)) struct ethosu_driver* ethosu_reserve_driver_ex(
    uint32_t,
    uint32_t) {
  return ethosu_reserve_driver();
}

namespace executorch {
namespace backends {
namespace arm {
namespace {

#ifdef ETHOSU_PERSISTENT_REGION
constexpr int ETHOSU_NUM_REGIONS = 6;
#else
constexpr int ETHOSU_NUM_REGIONS = 5;
constexpr int ETHOSU_NUM_BASE_ADDRS = 3;
#endif

/** Resolves shared scratch or allocates a separate temporary I/O region. */
Result<char*> prepare_copy_region(
    MemoryAllocator* temp_allocator,
    IOPlacement io_placement,
    size_t region_size,
    char* ethosu_scratch,
    const char* region_kind) {
  if (io_placement == IOPlacement::INSIDE_SCRATCH) {
    return ethosu_scratch;
  }
  if (region_size == 0) {
    return static_cast<char*>(nullptr);
  }

  char* region_base = static_cast<char*>(
      temp_allocator->allocate(region_size, ETHOSU_SCRATCH_ALIGNMENT));
  if (region_base != nullptr) {
    return region_base;
  }
  ET_LOG(
      Error,
      "Failed to allocate %s region buffer of %zu bytes from temp_allocator",
      region_kind,
      region_size);
  return Error::MemoryAllocationFailed;
}

} // namespace

// cppcheck-suppress unusedFunction
executorch::runtime::Error platform_init(
    executorch::runtime::ArrayRef<executorch::runtime::CompileSpec> /*specs*/,
    executorch::runtime::MemoryAllocator* /*allocator*/,
    ExecutionHandle* /*handle*/) {
  return executorch::runtime::Error::Ok;
}

// cppcheck-suppress unusedFunction
void platform_destroy(PlatformState* /*state*/) {}

// cppcheck-suppress unusedFunction
bool needs_scratch_allocation() {
  return true;
}

// cppcheck-suppress unusedFunction
Error platform_execute(
    BackendExecutionContext& context,
    [[maybe_unused]] const ExecutionHandle* execution_handle,
    IOPlan& io_plan) {
#if defined(ET_EVENT_TRACER_ENABLED)
  auto* event_tracer = context.event_tracer();
#endif
  const VelaHandles& handles = *io_plan.handles;
  MemoryAllocator* temp_allocator = context.get_temp_allocator();
  char* ethosu_scratch = reinterpret_cast<char*>(io_plan.scratch.data());
  if (handles.scratch_data_size > 0 && ethosu_scratch == nullptr) {
    ET_LOG(Error, "Ethos-U scratch buffer is missing");
    return Error::InvalidState;
  }
  if (io_plan.needs_input_copy) {
    auto input_copy_region = prepare_copy_region(
        temp_allocator,
        io_plan.placement,
        io_plan.input_region_size,
        ethosu_scratch,
        "input");
    if (!input_copy_region.ok()) {
      return input_copy_region.error();
    }
    io_plan.input_region_base = *input_copy_region;
    EXECUTORCH_PROF_SCOPE(event_tracer, "EthosUBackend::input.copy()");
    Error status = copy_inputs(io_plan);
    if (status != Error::Ok) {
      return status;
    }
  }

  if (io_plan.needs_output_copy) {
    auto output_copy_region = prepare_copy_region(
        temp_allocator,
        io_plan.placement,
        io_plan.output_region_size,
        ethosu_scratch,
        "output");
    if (!output_copy_region.ok()) {
      return output_copy_region.error();
    }
    io_plan.contiguous_output_base = *output_copy_region;
  }
  ET_LOG(
      Debug,
      "Running program data:\n  cmd %p %zu\n  weight %p %zu\n  scratch %p %zu\n  fast scratch %p %zu\n",
      handles.cmd_data,
      handles.cmd_data_size,
      handles.weight_data,
      handles.weight_data_size,
      ethosu_scratch,
      handles.scratch_data_size,
      ethosu_fast_scratch,
      ethosu_fast_scratch_size);

  // Parse product config from command stream to reserve the correct driver
  uint32_t product, log2_macs;
  // The weak fallback below always returns 0, but some builds replace it
  // with a real driver implementation that can return an error code.
  const int product_config_status = ethosu_get_product_config_from_cop_data(
      handles.cmd_data, handles.cmd_data_size, &product, &log2_macs);
  if (product_config_status != 0) { // cppcheck-suppress knownConditionTrueFalse
    ET_LOG(Error, "Failed to parse product config from command stream");
    return Error::InvalidProgram;
  }

  // Allocate driver handle and synchronously invoke driver
  auto driver =
      std::unique_ptr<ethosu_driver, decltype(&ethosu_release_driver)>(
          ethosu_reserve_driver_ex(product, log2_macs), ethosu_release_driver);
  if (driver == nullptr) {
    ET_LOG(Error, "ethosu_reserve_driver_ex failed");
    return Error::InvalidState;
  }

  uint64_t region_bases[ETHOSU_NUM_REGIONS] = {
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>((handles.weight_data))),
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ethosu_scratch)),
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ethosu_fast_scratch)),
      static_cast<uint64_t>(
          reinterpret_cast<uintptr_t>(io_plan.input_region_base)),
      static_cast<uint64_t>(
          reinterpret_cast<uintptr_t>(io_plan.contiguous_output_base)),
#ifdef ETHOSU_PERSISTENT_REGION
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
          execution_handle ? execution_handle->persistent_region : nullptr)),
#endif
  };
  size_t region_sizes[ETHOSU_NUM_REGIONS] = {
      handles.weight_data_size,
      handles.scratch_data_size,
      ethosu_fast_scratch_size,
      io_plan.input_region_size,
      io_plan.output_region_size,
#ifdef ETHOSU_PERSISTENT_REGION
      handles.persistent_data_size,
#endif
  };
#ifdef ETHOSU_PERSISTENT_REGION
  // Persistent state uses Vela region 5, after the optional I/O regions 3-4.
  constexpr int num_regions = ETHOSU_NUM_REGIONS;
#else
  const int num_regions = io_plan.placement == IOPlacement::OUTSIDE_SCRATCH
      ? ETHOSU_NUM_REGIONS
      : ETHOSU_NUM_BASE_ADDRS;
#endif
  EthosUInvocationContext invocation_context{};
  invocation_context.delegate_handle =
      reinterpret_cast<uintptr_t>(execution_handle);
#if defined(ET_EVENT_TRACER_ENABLED)
  invocation_context.event_tracer = event_tracer;
#endif
  int result;
  {
    EXECUTORCH_PROF_SCOPE(event_tracer, "EthosUBackend::driver.invoke()");
    result = ethosu_invoke_v3(
        driver.get(),
        static_cast<const void*>(handles.cmd_data),
        handles.cmd_data_size,
        region_bases,
        region_sizes,
        num_regions, /* fixed array of pointers to binary interface*/
        &invocation_context);
  }

  if (result != 0) {
    ET_LOG(Error, "Ethos-U invocation failed error (%d)", result);
    return Error::InvalidProgram;
  }

  if (io_plan.needs_output_copy) {
    EXECUTORCH_PROF_SCOPE(event_tracer, "EthosUBackend::output.copy()");
    return copy_outputs(io_plan);
  }
  return Error::Ok;
}

} // namespace arm
} // namespace backends
} // namespace executorch
