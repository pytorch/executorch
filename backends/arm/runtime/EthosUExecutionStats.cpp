/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <array>

#include <ethosu_driver.h>
#include <pmu_ethosu.h>

#include <executorch/backends/arm/runtime/EthosUBackend_Pmu.h>
#include <executorch/backends/arm/runtime/EthosUExecutionStats.h>

#if defined(ET_EVENT_TRACER_ENABLED)
namespace {
void write_le32(uint8_t* destination, uint32_t value) {
  for (size_t byte = 0; byte < 4; ++byte) {
    destination[byte] = static_cast<uint8_t>(value >> (8 * byte));
  }
}

void write_le64(uint8_t* destination, uint64_t value) {
  for (size_t byte = 0; byte < 8; ++byte) {
    destination[byte] = static_cast<uint8_t>(value >> (8 * byte));
  }
}
} // namespace
#endif

namespace executorch::backends::arm {
namespace {
std::atomic<EthosUStatsBuffer*> capture_buffer{nullptr};

std::array<ethosu_pmu_event_type, 4> u55_u65_events = {
    ETHOSU_PMU_AXI0_RD_DATA_BEAT_RECEIVED,
    ETHOSU_PMU_AXI1_RD_DATA_BEAT_RECEIVED,
    ETHOSU_PMU_AXI0_WR_DATA_BEAT_WRITTEN,
    ETHOSU_PMU_NPU_IDLE,
};

std::array<ethosu_pmu_event_type, kMaxPmuEventCounters> u85_events = {
    ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED,
    ETHOSU_PMU_SRAM_WR_DATA_BEAT_WRITTEN,
    ETHOSU_PMU_EXT_RD_DATA_BEAT_RECEIVED,
    ETHOSU_PMU_EXT_WR_DATA_BEAT_WRITTEN,
    ETHOSU_PMU_NPU_IDLE,
    ETHOSU_PMU_MAC_ACTIVE,
    ETHOSU_PMU_WD_ACTIVE,
    ETHOSU_PMU_NO_EVENT,
};

template <size_t N>
void copy_pmu_events(
    std::array<ethosu_pmu_event_type, N>& destination,
    executorch::runtime::ArrayRef<ethosu_pmu_event_type> events) {
  destination.fill(ETHOSU_PMU_NO_EVENT);
  if (!events.empty()) {
    std::copy_n(events.begin(), events.size(), destination.begin());
  }
}
} // namespace

// cppcheck-suppress unusedFunction
EthosUStatsBuffer* set_ethosu_stats_buffer(EthosUStatsBuffer* buffer) noexcept {
  return capture_buffer.exchange(buffer, std::memory_order_acq_rel);
}

// cppcheck-suppress unusedFunction
bool ethosu_execution_metrics_enabled() noexcept {
  return true;
}

// cppcheck-suppress unusedFunction
executorch::runtime::Error set_ethosu55_65_pmu_events(
    executorch::runtime::ArrayRef<ethosu_pmu_event_type> events) noexcept {
  using executorch::runtime::Error;
  if (events.size() > u55_u65_events.size()) {
    return Error::InvalidArgument;
  }
  copy_pmu_events(u55_u65_events, events);
  return Error::Ok;
}

// cppcheck-suppress unusedFunction
executorch::runtime::Error set_ethosu85_pmu_events(
    executorch::runtime::ArrayRef<ethosu_pmu_event_type> events) noexcept {
  using executorch::runtime::Error;
  if (events.size() > u85_events.size()) {
    return Error::InvalidArgument;
  }
  copy_pmu_events(u85_events, events);
  return Error::Ok;
}

} // namespace executorch::backends::arm

using executorch::backends::arm::EthosUInvocationContext;

extern "C" {
void __attribute__((weak)) EthosUBackend_npu_begin() {}

void __attribute__((weak)) EthosUBackend_npu_end() {}

void ethosu_inference_begin(struct ethosu_driver* drv, void* user_arg) {
  auto* invocation = static_cast<EthosUInvocationContext*>(user_arg);
  if (invocation == nullptr) {
    return;
  }

  invocation->counter_count = 0;
  invocation->buffer =
      executorch::backends::arm::capture_buffer.load(std::memory_order_acquire);
  if (invocation->buffer != nullptr
#if defined(ET_EVENT_TRACER_ENABLED)
      || invocation->event_tracer != nullptr
#endif
  ) {
    const ethosu_pmu_event_type* events = nullptr;
    uint32_t event_count = 0;
    if (drv->dev.caps.product == ETHOSU_PRODUCT_U55 ||
        drv->dev.caps.product == ETHOSU_PRODUCT_U65) {
      events = executorch::backends::arm::u55_u65_events.data();
      event_count = executorch::backends::arm::u55_u65_events.size();
    } else if (drv->dev.caps.product == ETHOSU_PRODUCT_U85) {
      events = executorch::backends::arm::u85_events.data();
      event_count = executorch::backends::arm::u85_events.size();
    }

    if (events != nullptr) {
      invocation->counter_count =
          std::min(event_count, ETHOSU_PMU_Get_NumEventCountersForDrv(drv));
    }
    if (events != nullptr && invocation->counter_count != 0) {
      ETHOSU_PMU_Enable(drv);
      ETHOSU_PMU_PMCCNTR_CFG_Set_Stop_Event(drv, ETHOSU_PMU_NPU_IDLE);
      ETHOSU_PMU_PMCCNTR_CFG_Set_Start_Event(drv, ETHOSU_PMU_NPU_ACTIVE);
      uint32_t event_mask = 0;
      for (uint32_t slot = 0; slot < invocation->counter_count; ++slot) {
        const auto event = events[slot];
        ETHOSU_PMU_Set_EVTYPER(drv, slot, event);
        if (event != ETHOSU_PMU_NO_EVENT) {
          event_mask |= 1u << slot;
        }
      }
      ETHOSU_PMU_CNTR_Disable(drv, (1u << invocation->counter_count) - 1u);
      ETHOSU_PMU_CNTR_Enable(drv, event_mask | ETHOSU_PMU_CCNT_Msk);
      ETHOSU_PMU_CYCCNT_Reset(drv);
      ETHOSU_PMU_EVCNTR_ALL_Reset(drv);
      ETHOSU_PMU_Set_CNTR_OVS(drv, event_mask | ETHOSU_PMU_CCNT_Msk);
#if defined(ET_EVENT_TRACER_ENABLED)
      if (invocation->event_tracer != nullptr) {
        invocation->cycle_event =
            invocation->event_tracer->start_profiling_delegate(
                executorch::backends::arm::kEthosUCycleEventName,
                executorch::runtime::kUnsetDelegateDebugIntId);
      }
#endif
    }
  }

  // Call callback
  EthosUBackend_npu_begin();
}

void ethosu_inference_end(struct ethosu_driver* drv, void* user_arg) {
  const auto* invocation = static_cast<EthosUInvocationContext*>(user_arg);
  if (invocation == nullptr) {
    return;
  }

  if (invocation->counter_count != 0) {
    executorch::backends::arm::EthosUExecutionRecord record{};
    record.delegate_handle = invocation->delegate_handle;
    record.success = drv->job.result == ETHOSU_JOB_RESULT_OK;
    record.stats.device_identifier = reinterpret_cast<uintptr_t>(drv->dev.reg);
    record.stats.product = drv->dev.caps.product;
    record.stats.macs_per_cycle = 1u << drv->dev.caps.log2_macs;
    record.stats.cycle_count = ETHOSU_PMU_Get_CCNTR(drv);
    record.stats.cycle_count_valid = true;
    const uint32_t overflow_status = ETHOSU_PMU_Get_CNTR_OVS(drv);
    record.stats.cycle_overflow = (overflow_status & ETHOSU_PMU_CCNT_Msk) != 0;
    for (uint32_t slot = 0; slot < invocation->counter_count; ++slot) {
      const auto event = ETHOSU_PMU_Get_EVTYPER(drv, slot);
      if (event == ETHOSU_PMU_NO_EVENT) {
        continue;
      }
      record.stats.counters[record.stats.counter_count++] = {
          static_cast<uint32_t>(event),
          ETHOSU_PMU_Get_EVCNTR(drv, slot),
          (overflow_status & (1u << slot)) != 0};
    }
    ETHOSU_PMU_Disable(drv);
    EthosUBackend_npu_end();
#if defined(ET_EVENT_TRACER_ENABLED)
    if (invocation->event_tracer != nullptr) {
      // Encode metadata in format described in EthosUExecutionStats.h
      uint8_t metadata[executorch::backends::arm::kEthosUCycleMetadataMaxSize]{
          'E', 'U', 'C', '1'};
      write_le32(metadata + 4, record.stats.product);
      write_le32(metadata + 8, record.stats.macs_per_cycle);
      write_le32(metadata + 12, static_cast<uint32_t>(record.stats.platform));
      const uint32_t flags =
          (record.stats.cycle_overflow ? 1u : 0u) | (record.success ? 2u : 0u);
      write_le32(metadata + 16, flags);
      write_le64(metadata + 20, record.stats.device_identifier);
      write_le64(metadata + 28, record.stats.cycle_count);
      write_le32(metadata + 36, record.stats.counter_count);
      for (uint32_t slot = 0; slot < record.stats.counter_count; ++slot) {
        const size_t offset =
            executorch::backends::arm::kEthosUCycleMetadataHeaderSize +
            slot * executorch::backends::arm::kEthosUCycleCounterMetadataSize;
        const auto& counter = record.stats.counters[slot];
        write_le32(metadata + offset, counter.event_id);
        write_le64(metadata + offset + 4, counter.value);
        write_le32(metadata + offset + 12, counter.overflow ? 1u : 0u);
      }
      const size_t metadata_size =
          executorch::backends::arm::kEthosUCycleMetadataHeaderSize +
          record.stats.counter_count *
              executorch::backends::arm::kEthosUCycleCounterMetadataSize;
      invocation->event_tracer->end_profiling_delegate(
          invocation->cycle_event, metadata, metadata_size);
    }
#endif
    if (invocation->buffer != nullptr) {
      invocation->buffer->append(record);
    }
    return;
  }
  EthosUBackend_npu_end();
}
}
