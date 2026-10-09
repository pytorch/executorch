/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Arm backend for Ethos-U baremetal driver stack, this relies on the
 * ethos-u-core-driver for hardware interaction.
 */

#include <cstdint>
#include <cstring>
#include <memory>

#include <ethosu_driver.h>

#include <executorch/backends/arm/runtime/EthosUBackend_Internal.h>
#include <executorch/runtime/core/error.h>

using executorch::runtime::BackendExecutionContext;
using executorch::runtime::Error;
using executorch::runtime::Span;

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

// Overridable memcpy for copying outputs from scratch.
// Default (weak) implementation in EthosUBackend_IoMemcpy.cpp does
// std::memcpy. Firmware targets can supply a strong override (e.g. routing
// through a DMA engine) to reduce CPU memcpy load on the host MCU.
extern "C" void arm_ethos_io_memcpy(void* dst, const void* src, size_t size);

namespace executorch {
namespace backends {
namespace arm {

struct PlatformState {};

executorch::runtime::Error platform_init(
    executorch::runtime::ArrayRef<executorch::runtime::CompileSpec> /*specs*/,
    executorch::runtime::MemoryAllocator* /*allocator*/,
    ExecutionHandle* /*handle*/) {
  return executorch::runtime::Error::Ok;
}

void platform_destroy(PlatformState* /*state*/) {}

bool needs_scratch_allocation() {
  return true;
}

Error platform_execute(
    BackendExecutionContext& /*context*/,
    const ExecutionHandle* /*execution_handle*/,
    const VelaHandles& handles,
    int input_count,
    int output_count,
    Span<executorch::runtime::EValue*> args,
    char* ethosu_scratch) {
  if (handles.scratch_data_size > 0 && ethosu_scratch == nullptr) {
    ET_LOG(Error, "Ethos-U scratch buffer is missing");
    return Error::InvalidState;
  }

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

  // Ethos-U low level driver expected order for Ethos U-55, we have
  // constant weight data, then scratch (which contains input and output)
  // scratch is written above in this function.
  uint64_t bases[ETHOSU_NUM_BASE_ADDRS] = {
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>((handles.weight_data))),
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ethosu_scratch)),
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ethosu_fast_scratch))};
  size_t bases_size[ETHOSU_NUM_BASE_ADDRS] = {
      handles.weight_data_size,
      handles.scratch_data_size,
      ethosu_fast_scratch_size};
  int result = ethosu_invoke_v3(
      driver.get(),
      static_cast<const void*>(handles.cmd_data),
      handles.cmd_data_size,
      bases,
      bases_size,
      ETHOSU_NUM_BASE_ADDRS, /* fixed array of pointers to binary interface*/
      nullptr);

  if (result != 0) {
    ET_LOG(Error, "Ethos-U invocation failed error (%d)", result);
    return Error::InvalidProgram;
  }

  // Write outputs from scratch into EValue pointers.
  for (int i = 0; i < output_count; i++) {
    const char* output_addr = ethosu_scratch + handles.outputs->io[i].offset;
    auto tensor_out = args[input_count + i]->toTensor();
    const size_t tensor_bytes = tensor_out.nbytes();

    // Routed through arm_ethos_io_memcpy so firmware can DMA-accelerate.
#if defined(ET_ARM_ETHOSU_PROFILE_IO_COPIES)
    EthosUBackend_output_memcpy(tensor_bytes);
#endif
    arm_ethos_io_memcpy(
        tensor_out.mutable_data_ptr<char>(), output_addr, tensor_bytes);
  }
  return Error::Ok;
}

} // namespace arm
} // namespace backends
} // namespace executorch
