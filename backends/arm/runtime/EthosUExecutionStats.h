/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

#include <pmu_ethosu.h>

#include <executorch/runtime/core/array_ref.h>
#include <executorch/runtime/core/error.h>

// Optional runner hooks. The stats library provides weak no-op definitions.
extern "C" {
void EthosUBackend_npu_begin();
void EthosUBackend_npu_end();
}

namespace executorch::backends::arm {

// Containers of Ethos-U metrics.
// An EthosUStatsBuffer contains many EthosUExecutionRecords.
// An EthosUExecutionRecord contains delegate id and status, and an
// EthosUExecutionStats, containing Ethos-U specific metrics.
struct EthosUExecutionStats;
struct EthosUExecutionRecord;
struct EthosUStatsBuffer;

// Public API of the ethosu_execution_stats library.
// Register runner-owned storage for subsequent invocations. Pass nullptr to
// stop capture; replace the buffer only when no invocations are running.
EthosUStatsBuffer* set_ethosu_stats_buffer(EthosUStatsBuffer* buffer) noexcept;
// Returns true when this library is linked.
bool ethosu_execution_metrics_enabled() noexcept;

// Configure counter slots before inference. U55/U65 share four slots; U85 has
// eight. Short lists are padded with ETHOSU_PMU_NO_EVENT; longer lists are
// rejected. Pass an empty list to disable event counters for that target.
// These functions only configure events; collection requires an event tracer
// or a registered EthosUStatsBuffer. Do not call during inference.
executorch::runtime::Error set_ethosu55_65_pmu_events(
    executorch::runtime::ArrayRef<ethosu_pmu_event_type> events) noexcept;
executorch::runtime::Error set_ethosu85_pmu_events(
    executorch::runtime::ArrayRef<ethosu_pmu_event_type> events) noexcept;

constexpr size_t kMaxPmuEventCounters = 8;

constexpr char kEthosUCycleEventName[] = "ethosu.pmu.counters";
// Experimental format; layout and fields may change without compatibility.
// ETDump delegate_debug_metadata starts with "EUC" and ASCII version '1'
// [0:4], followed by little-endian product [4:8], MACs per cycle [8:12],
// EthosUPlatform [12:16], flags [16:20] (bit 0: cycle overflow,
// bit 1: invocation succeeded), register base [20:28], NPU cycle count
// [28:36], and counter count [36:40].
// Each active counter then stores a symbolic event ID (4 bytes), count
// (8 bytes), and overflow (4 bytes). NO_EVENT slots are omitted.
constexpr size_t kEthosUCycleMetadataHeaderSize = 40;
constexpr size_t kEthosUCycleCounterMetadataSize = 16;
constexpr size_t kEthosUCycleMetadataMaxSize = kEthosUCycleMetadataHeaderSize +
    kMaxPmuEventCounters * kEthosUCycleCounterMetadataSize;

enum class EthosUPlatform : uint8_t {
  CoreDriver,
};

struct EthosUPmuCounter {
  uint32_t event_id = 0;
  uint64_t value = 0;
  bool overflow = false;
};

struct EthosUExecutionStats {
  // Register base address of the physical NPU that produced this record.
  uintptr_t device_identifier = 0;
  // Ethos-U product identifier for the NPU that produced this record.
  uint32_t product = 0;
  // Number of MACs per cycle supported by the NPU.
  uint32_t macs_per_cycle = 0;
  // Platform that produced this record.
  EthosUPlatform platform = EthosUPlatform::CoreDriver;
  // NPU PMU cycle count for this execution, when available.
  uint64_t cycle_count = 0;
  // Whether cycle_count was measured; zero can be a valid measurement.
  bool cycle_count_valid = false;
  // Whether the cycle counter overflowed, if overflow status is available.
  bool cycle_overflow = false;
  // Number of populated entries at the start of counters.
  uint32_t counter_count = 0;
  // PMU event IDs, counts, and overflow indications for this execution.
  std::array<EthosUPmuCounter, kMaxPmuEventCounters> counters{};

  void reset() {
    *this = {};
  }
};

struct EthosUExecutionRecord {
  EthosUExecutionStats stats;
  uintptr_t delegate_handle = 0;
  bool success = false;
};

// Appends from concurrent invocations are synchronized. Read the records only
// after those invocations have completed.
struct EthosUStatsBuffer {
  // Records are added at the start of the supplied array.
  EthosUExecutionRecord* records = nullptr;
  size_t capacity = 0;
  size_t size = 0;
  bool overflow = false;

  constexpr EthosUStatsBuffer() = default;
  constexpr EthosUStatsBuffer(
      EthosUExecutionRecord* records,
      size_t capacity,
      bool overflow = false) noexcept
      : records(records), capacity(capacity), overflow(overflow) {}

  void append(const EthosUExecutionRecord& record) noexcept {
    lock();
    if (size >= capacity) {
      overflow = true;
    } else {
      records[size++] = record;
    }
    unlock();
  }

  // Read records or reset this buffer only after all publishers have completed.
  bool reset() noexcept {
    lock();
    const bool had_overflow = overflow;
    size = 0;
    overflow = false;
    unlock();
    return had_overflow;
  }

 private:
  void lock() noexcept {
    while (append_lock_.test_and_set(std::memory_order_acquire)) {
    }
  }

  void unlock() noexcept {
    append_lock_.clear(std::memory_order_release);
  }

  std::atomic_flag append_lock_ = ATOMIC_FLAG_INIT;
};

} // namespace executorch::backends::arm
