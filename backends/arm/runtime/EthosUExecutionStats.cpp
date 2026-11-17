/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <array>
#if defined(EXECUTORCH_BUILD_ARM_ETHOSU_LINUX)
#include <executorch/runtime/platform/log.h>
#include <exception>
#include <vector>
#endif

#include <executorch/backends/arm/runtime/EthosUExecutionStats.h>
#include <executorch/backends/arm/runtime/EthosUExecutionStats_Internal.h>

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
using executorch::runtime::ArrayRef;
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
#if defined(EXECUTORCH_BUILD_ARM_ETHOSU_LINUX)
  for (auto event : events) {
    if (!ethosu_linux_pmu_event_supported(1, event)) {
      return Error::InvalidArgument;
    }
  }
#endif
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
#if defined(EXECUTORCH_BUILD_ARM_ETHOSU_LINUX)
  for (auto event : events) {
    if (!ethosu_linux_pmu_event_supported(2, event)) {
      return Error::InvalidArgument;
    }
  }
#endif
  copy_pmu_events(u85_events, events);
  return Error::Ok;
}

ArrayRef<ethosu_pmu_event_type> ethosu_prepare_invocation(
    EthosUInvocationContext& invocation,
    uint32_t product) {
  invocation.pmu_enabled = false;
  invocation.buffer = capture_buffer.load(std::memory_order_acquire);
  if (invocation.buffer == nullptr
#if defined(ET_EVENT_TRACER_ENABLED)
      && invocation.event_tracer == nullptr
#endif
  ) {
    return {};
  }
  if (product == 0 || product == 1) {
    return {u55_u65_events.data(), u55_u65_events.size()};
  }
  if (product == 2) {
    return {u85_events.data(), u85_events.size()};
  }
  return {};
}

// cppcheck-suppress constParameterReference
void ethosu_execution_begin(EthosUInvocationContext& invocation) {
#if defined(ET_EVENT_TRACER_ENABLED)
  if (invocation.pmu_enabled && invocation.event_tracer != nullptr) {
    invocation.cycle_event = invocation.event_tracer->start_profiling_delegate(
        kEthosUCycleEventName, executorch::runtime::kUnsetDelegateDebugIntId);
  }
#endif
  EthosUBackend_npu_begin();
}

void ethosu_execution_end(
    const EthosUInvocationContext& invocation,
    const EthosUExecutionRecord* result) {
  EthosUBackend_npu_end();
#if defined(ET_EVENT_TRACER_ENABLED)
  if (invocation.pmu_enabled && invocation.event_tracer != nullptr) {
    if (result == nullptr) {
      invocation.event_tracer->end_profiling_delegate(
          invocation.cycle_event, nullptr, 0);
      return;
    }
    const auto& record = *result;
    size_t metadata_size = kEthosUCycleMetadataHeaderSize +
        record.stats.counter_count * kEthosUCycleCounterMetadataSize;
#if defined(EXECUTORCH_BUILD_ARM_ETHOSU_LINUX)
    std::vector<uint8_t> storage;
    try {
      storage.resize(metadata_size + 4 + record.stats.device_identifier.size());
    } catch (const std::exception& error) {
      ET_LOG(Debug, "Failed to allocate Ethos-U metadata: %s", error.what());
      invocation.event_tracer->end_profiling_delegate(
          invocation.cycle_event, nullptr, 0);
      return;
    }
    uint8_t* metadata = storage.data();
#else
    uint8_t metadata[kEthosUCycleMetadataMaxSize]{};
#endif
    std::copy_n("EUC1", 4, metadata);
    write_le32(metadata + 4, record.stats.product);
    write_le32(metadata + 8, record.stats.macs_per_cycle);
    write_le32(metadata + 12, static_cast<uint32_t>(record.stats.platform));
    const uint32_t flags =
        (record.stats.cycle_overflow ? 1u : 0u) | (record.success ? 2u : 0u);
    write_le32(metadata + 16, flags);
#if !defined(EXECUTORCH_BUILD_ARM_ETHOSU_LINUX)
    write_le64(metadata + 20, record.stats.device_identifier);
#endif
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
#if defined(EXECUTORCH_BUILD_ARM_ETHOSU_LINUX)
    const auto& path = record.stats.device_identifier;
    write_le32(metadata + metadata_size, path.size());
    std::copy(path.begin(), path.end(), metadata + metadata_size + 4);
    metadata_size = storage.size();
#endif
    invocation.event_tracer->end_profiling_delegate(
        invocation.cycle_event, metadata, metadata_size);
  }
#endif
  if (result != nullptr && invocation.buffer != nullptr) {
#if defined(EXECUTORCH_BUILD_ARM_ETHOSU_LINUX)
    try {
      invocation.buffer->append(*result);
    } catch (const std::exception& error) {
      ET_LOG(Debug, "Failed to capture Ethos-U stats: %s", error.what());
    }
#else
    invocation.buffer->append(*result);
#endif
  }
}

} // namespace executorch::backends::arm

extern "C" {
void __attribute__((weak)) EthosUBackend_npu_begin() {}
void __attribute__((weak)) EthosUBackend_npu_end() {}
}
