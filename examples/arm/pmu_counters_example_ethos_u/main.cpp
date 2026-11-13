/* Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/arm/runtime/EthosUExecutionStats.h>
#include <executorch/devtools/etdump/etdump_flatcc.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/third-party/flatcc/include/flatcc/portable/pbase64.h>
#include <stdio.h>
#include <cinttypes>
#include <cstring>
#include <vector>

#include "arm_memory_allocator.h"

#include "model_pte.h"

using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::backends::arm::EthosUExecutionRecord;
using executorch::backends::arm::EthosUStatsBuffer;
using executorch::backends::arm::set_ethosu55_65_pmu_events;
using executorch::backends::arm::set_ethosu85_pmu_events;
using executorch::backends::arm::set_ethosu_stats_buffer;
using executorch::extension::BufferDataLoader;
using executorch::runtime::Error;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;

alignas(16) __attribute__((section("input_data_sec"))) unsigned char method_pool
    [ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE];
alignas(16)
    __attribute__((section(".bss.tensor_arena"))) unsigned char temp_pool
        [ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE];

[[maybe_unused]] void et_pal_init() {
#if defined(__PMU_PRESENT) && (__PMU_PRESENT == 1U)
  ARM_PMU_Enable();
  DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
  ARM_PMU_CYCCNT_Reset();
  ARM_PMU_CNTR_Enable(PMU_CNTENSET_CCNTR_ENABLE_Msk);
#endif
}

[[maybe_unused]] et_timestamp_t et_pal_current_ticks() {
#if defined(__PMU_PRESENT) && (__PMU_PRESENT == 1U)
  return ARM_PMU_Get_CCNTR();
#else
  return 0;
#endif
}

// ETDump timestamps are cycles; the target CPU frequency is not assumed.
[[maybe_unused]] et_tick_ratio_t et_pal_ticks_to_ns_multiplier() {
  return {1, 1};
}

[[maybe_unused]] void et_pal_emit_log_message(
    ET_UNUSED et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    ET_UNUSED const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  fprintf(
      stderr,
      "%c [executorch:%s:%lu] %s\n",
      level,
      filename,
      static_cast<unsigned long>(line),
      message);
}

int main() {
  printf("Ethos-U PMU example started\n");
  executorch::runtime::runtime_init();

  BufferDataLoader loader(model_pte, sizeof(model_pte));
  Result<Program> program = Program::load(&loader);
  ET_CHECK_MSG(program.ok(), "Program::load failed: 0x%x", program.error());

  const auto method_name = program->get_method_name(0);
  ET_CHECK_MSG(method_name.ok(), "Program has no methods");
  const auto method_meta = program->method_meta(*method_name);
  ET_CHECK_MSG(method_meta.ok(), "Could not load method metadata");

  ArmMemoryAllocator method_allocator(sizeof(method_pool), method_pool);
  ArmMemoryAllocator temp_allocator(sizeof(temp_pool), temp_pool);

  std::vector<Span<uint8_t>> planned_spans;
  planned_spans.reserve(method_meta->num_memory_planned_buffers());
  for (size_t i = 0; i < method_meta->num_memory_planned_buffers(); ++i) {
    const size_t size = method_meta->memory_planned_buffer_size(i).get();
    auto* buffer = static_cast<uint8_t*>(method_allocator.allocate(size, 16));
    ET_CHECK_MSG(buffer != nullptr, "Could not allocate planned buffer %lu", i);
    planned_spans.push_back({buffer, size});
  }

  HierarchicalAllocator planned_memory(
      {planned_spans.data(), planned_spans.size()});
  MemoryManager memory_manager(
      &method_allocator, &planned_memory, &temp_allocator);

  // Passing a tracer captures PMU records as ethosu.pmu.counters metadata.
  constexpr size_t etdump_buffer_size = 512 * 1024;
  auto* etdump_buffer =
      static_cast<uint8_t*>(method_allocator.allocate(etdump_buffer_size, 64));
  ET_CHECK_MSG(etdump_buffer != nullptr, "Could not allocate ETDump buffer");
  executorch::etdump::ETDumpGen etdump({etdump_buffer, etdump_buffer_size});
  Result<Method> method =
      program->load_method(*method_name, &memory_manager, &etdump);
  ET_CHECK_MSG(method.ok(), "Could not load method: 0x%x", method.error());

  for (size_t i = 0; i < method_meta->num_inputs(); ++i) {
    auto tensor_meta = method_meta->input_tensor_meta(i);
    ET_CHECK_MSG(tensor_meta.ok(), "Input %zu must be a tensor", i);
    void* input_buffer = method_allocator.allocate(tensor_meta->nbytes(), 16);
    ET_CHECK_MSG(
        input_buffer != nullptr,
        "Could not allocate input %lu",
        static_cast<unsigned long>(i));
    TensorImpl input_impl(
        tensor_meta->scalar_type(),
        tensor_meta->sizes().size(),
        const_cast<TensorImpl::SizesType*>(tensor_meta->sizes().data()),
        input_buffer,
        const_cast<TensorImpl::DimOrderType*>(tensor_meta->dim_order().data()));
    Tensor input(&input_impl);
    // Use zero-filled placeholder data for every model input tensor.
    std::memset(input_buffer, 0, tensor_meta->nbytes());
    ET_CHECK_MSG(
        method->set_input(input, i) == Error::Ok,
        "Could not prepare input %lu",
        static_cast<unsigned long>(i));
  }

  // Configure each product family before inference; only the active NPU is
  // used.
  constexpr ethosu_pmu_event_type events[] = {
      ETHOSU_PMU_MAC_ACTIVE, ETHOSU_PMU_NPU_IDLE};
  ET_CHECK_MSG(
      set_ethosu55_65_pmu_events(events) == Error::Ok,
      "Invalid U55/U65 events");
  ET_CHECK_MSG(
      set_ethosu85_pmu_events(events) == Error::Ok, "Invalid U85 events");

  // The buffer independently captures the same invocations without a tracer.
  // Set buffer size depending on expected number of invocations. This example
  // picks 16.
  EthosUExecutionRecord records[16];
  EthosUStatsBuffer stats_buffer(records, 16);
  set_ethosu_stats_buffer(&stats_buffer);
  printf("Starting inference\n");
  ET_CHECK_MSG(method->execute() == Error::Ok, "Inference failed");
  printf("Model executed successfully.\n");

  // Read records only after execute() has completed, then unregister storage.
  // The backend has no ownership of the buffer, so setting it to nullptr
  // doesn't nullify the buffer, it unregisters.
  set_ethosu_stats_buffer(nullptr);
  ET_CHECK_MSG(stats_buffer.size > 0, "No Ethos-U invocations captured");
  ET_CHECK_MSG(!stats_buffer.overflow, "Increase PMU record capacity");
  for (size_t i = 0; i < stats_buffer.size; ++i) {
    const auto& record = records[i];
    ET_CHECK_MSG(record.success, "Ethos-U invocation failed");
    const auto& stats = record.stats;
    printf("Delegate %" PRIuPTR "\n", record.delegate_handle);
    if (stats.cycle_count_valid) {
      printf(
          "  cycles: %lu overflow=%u\n",
          static_cast<unsigned long>(stats.cycle_count),
          unsigned(stats.cycle_overflow));
    }
    for (size_t j = 0; j < stats.counter_count; ++j) {
      const auto& counter = stats.counters[j];
      printf(
          "  event %" PRIu32 ": %lu overflow=%u\n",
          counter.event_id,
          static_cast<unsigned long>(counter.value),
          unsigned(counter.overflow));
    }
  }

  // run_fvp.sh extracts this base64 payload into etdump.bin.
  const auto dump = etdump.get_etdump_data();
  ET_CHECK_MSG(dump.buf != nullptr && dump.size > 0, "ETDump is empty");
  constexpr int mode = base64_enc_modifier_padding;
  size_t input_size = dump.size;
  size_t encoded_size = base64_encoded_size(input_size, mode);
  std::vector<char> encoded(encoded_size + 1);
  ET_CHECK_MSG(
      base64_encode(
          reinterpret_cast<uint8_t*>(encoded.data()),
          static_cast<const uint8_t*>(dump.buf),
          &encoded_size,
          &input_size,
          mode) == BASE64_EOK,
      "Could not encode ETDump");
  encoded[encoded_size] = '\0';
  printf("#[RUN THIS]\n");
  printf("echo \"%s\" | base64 -d >etdump.bin\n", encoded.data());
  printf("#[END]\n");

  printf("\04");
  return 0;
}
