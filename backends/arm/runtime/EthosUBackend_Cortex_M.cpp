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
#include <new>

#include <ethosu_driver.h>

#include <executorch/backends/arm/runtime/EthosUBackend_Internal.h>
#include <executorch/runtime/core/error.h>

using executorch::runtime::BackendExecutionContext;
using executorch::runtime::Error;
using executorch::runtime::Span;

namespace executorch {
namespace backends {
namespace arm {

struct PlatformState {};

PlatformState* platform_init(
    executorch::runtime::ArrayRef<executorch::runtime::CompileSpec> /*specs*/,
    executorch::runtime::MemoryAllocator* /*allocator*/) {
  return nullptr;
}

void platform_destroy(PlatformState* state) {
  delete state;
}

Error platform_execute(
    BackendExecutionContext& /*context*/,
    const ExecutionHandle* /*execution_handle*/,
    const VelaHandles& handles,
    int input_count,
    int output_count,
    Span<executorch::runtime::EValue*> args,
    char* ethosu_scratch) {
  // Allocate driver handle and synchronously invoke driver
  auto driver =
      std::unique_ptr<ethosu_driver, decltype(&ethosu_release_driver)>(
          ethosu_reserve_driver(), ethosu_release_driver);
  if (driver == nullptr) {
    ET_LOG(Error, "ethosu_reserve_driver failed");
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

  size_t tensor_bytes_total = 0;
  size_t io_bytes_total = 0;
  // Write outputs from scratch into EValue pointers
  for (int i = 0; i < output_count; i++) {
    int tensor_count = 1, io_count = 1;
    const char* output_addr = ethosu_scratch + handles.outputs->io[i].offset;
    // Process input EValue into scratch
    // Outputs are in the index immediately after inputs
    auto tensor_out = args[input_count + i]->toTensor();

    calculate_dimensions(
        tensor_out, &handles.outputs->io[i], &tensor_count, &io_count);

    size_t tensor_bytes = tensor_out.nbytes();
    size_t io_bytes = static_cast<size_t>(io_count) *
        static_cast<size_t>(handles.outputs->io[i].elem_size);

    if (tensor_bytes != io_bytes) {
      Error status = copy_with_layout_adjustment(
          handles.outputs->io[i], i, output_addr, tensor_out, tensor_bytes);
      if (status != Error::Ok) {
        return status;
      }
      io_bytes_total += tensor_bytes;
    } else {
      memcpy(
          tensor_out.mutable_data_ptr<char>(),
          static_cast<const char*>(output_addr),
          tensor_bytes);
      io_bytes_total += io_bytes;
    }

    // At times the topological order of the outputs may change.
    // Lets instead ensure that the sum of output bytes match.
    tensor_bytes_total += tensor_bytes;
  }
  if (tensor_bytes_total != io_bytes_total) {
    ET_LOG(Error, "Total output tensor sizes do not match");
    ET_LOG(
        Error,
        "Program expects %zu bytes but got %zu",
        io_bytes_total,
        tensor_bytes_total);
    return Error::InvalidProgram;
  }
  return Error::Ok;
}

} // namespace arm
} // namespace backends
} // namespace executorch
