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

namespace executorch::backends::arm {

// Containers of Ethos-U metrics.
// An EthosUStatsBuffer contains many EthosUExecutionRecords.
// An EthosUExecutionRecord contains delegate id and status, and an
// EthosUExecutionStats, containing Ethos-U specific metrics.
struct EthosUExecutionStats;
struct EthosUExecutionRecord;
struct EthosUStatsBuffer;

constexpr size_t kMaxPmuEventCounters = 8;

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
