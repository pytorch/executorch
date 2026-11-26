/* Copyright 2024-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <cinttypes>

#include "arm_perf_monitor.h"

using printf_size_t = unsigned long;

#ifdef ETHOSU
#include <executorch/backends/arm/runtime/EthosUBackend_Pmu.h>
#include <executorch/runtime/platform/log.h>
#if defined(ET_ARM_ETHOSU_PMU_STATS)
#include <executorch/backends/arm/runtime/EthosUExecutionStats.h>
#include <executorch/runtime/platform/assert.h>
#endif

namespace {

// Returns the Armv8.1-M PMU cycle counter; 0 on cores without it.
static inline uint64_t arm_pmu_cycles() {
#if defined(__PMU_PRESENT) && (__PMU_PRESENT == 1U)
  return ARM_PMU_Get_CCNTR();
#else
  return 0;
#endif
}

uint32_t ethosu_delegation_count = 0;
uint64_t ethosu_ArmCycleCountStart = 0;
uint64_t ethosu_ArmBackendExecuteCycleCountStart = 0;
uint64_t ethosu_ArmBackendExecuteCycleCount = 0;
uint64_t ethosu_ArmWhenNPURunCycleCountStart = 0;
uint64_t ethosu_ArmWhenNPURunCycleCount = 0;

#if defined(ET_ARM_ETHOSU_PMU_STATS)
using executorch::backends::arm::EthosUExecutionRecord;
using executorch::backends::arm::EthosUExecutionStats;
using executorch::backends::arm::EthosUStatsBuffer;

constexpr size_t kMaxPmuRecordsPerInference = 16;
constexpr size_t kMaxProfiledDevices = 1;

std::array<EthosUExecutionRecord, kMaxPmuRecordsPerInference> ethosu_pmuRecords;
EthosUStatsBuffer ethosu_pmuBuffer{
    ethosu_pmuRecords.data(),
    ethosu_pmuRecords.size()};

struct DevicePmuTotals {
  EthosUExecutionStats stats{};
  uint64_t invocations = 0;
  uint64_t failures = 0;
};
std::array<DevicePmuTotals, kMaxProfiledDevices> ethosu_deviceTotals;
size_t ethosu_deviceCount = 0;
bool ethosu_deviceCapacityExceeded = false;
bool ethosu_recordCapacityExceeded = false;
uint64_t ethosu_pmuCycleCount = 0;
#if defined(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
struct DelegatePmuTotals {
  uintptr_t handle = 0;
  DevicePmuTotals totals;
};
std::array<DelegatePmuTotals, ET_ARM_ETHOSU_MAX_PROFILED_DELEGATES>
    ethosu_delegatePmuTotals;
size_t ethosu_delegatePmuCount = 0;
bool ethosu_delegatePmuCapacityExceeded = false;
#endif

bool same_layout(
    const EthosUExecutionStats& left,
    const EthosUExecutionStats& right) {
  if (left.device_identifier != right.device_identifier ||
      left.product != right.product || left.platform != right.platform ||
      left.counter_count != right.counter_count) {
    return false;
  }
  for (uint32_t slot = 0; slot < left.counter_count; ++slot) {
    if (left.counters[slot].event_id != right.counters[slot].event_id) {
      return false;
    }
  }
  return true;
}

void accumulate_pmu_totals(
    DevicePmuTotals& totals,
    const EthosUExecutionRecord& record) {
  if (totals.invocations == 0) {
    totals.stats = record.stats;
    totals.stats.cycle_count = 0;
    totals.stats.cycle_overflow = false;
    for (uint32_t slot = 0; slot < totals.stats.counter_count; ++slot) {
      totals.stats.counters[slot].value = 0;
      totals.stats.counters[slot].overflow = false;
    }
  }
  ++totals.invocations;
  totals.failures += !record.success;
  totals.stats.cycle_count += record.stats.cycle_count;
  totals.stats.cycle_overflow |= record.stats.cycle_overflow;
  for (uint32_t slot = 0; slot < record.stats.counter_count; ++slot) {
    totals.stats.counters[slot].value += record.stats.counters[slot].value;
    totals.stats.counters[slot].overflow |=
        record.stats.counters[slot].overflow;
  }
}
#endif
#if defined(ET_ARM_ETHOSU_PROFILE_IO_COPIES)
struct IOCopyStats {
  uint64_t calls = 0;
  uint64_t bytes = 0;
};
IOCopyStats ethosu_inputCopyStats;
IOCopyStats ethosu_outputCopyStats;
#endif
#if defined(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
struct DelegateStats {
  const void* handle = nullptr;
  uint64_t backend_invocations = 0;
  uint64_t npu_invocations = 0;
  uint64_t pmu_cycles = 0;
#if defined(ET_ARM_ETHOSU_PROFILE_IO_COPIES)
  IOCopyStats input_copies;
  IOCopyStats output_copies;
#endif
};

std::array<DelegateStats, ET_ARM_ETHOSU_MAX_PROFILED_DELEGATES>
    ethosu_delegateStats;
size_t ethosu_delegateCount = 0;
DelegateStats* ethosu_activeDelegate = nullptr;
bool ethosu_delegateCapacityExceeded = false;

DelegateStats* get_delegate_stats(const void* handle) {
  for (size_t i = 0; i < ethosu_delegateCount; ++i) {
    if (ethosu_delegateStats[i].handle == handle) {
      return &ethosu_delegateStats[i];
    }
  }
  if (ethosu_delegateCount == ethosu_delegateStats.size()) {
    ethosu_delegateCapacityExceeded = true;
    return nullptr;
  }
  DelegateStats& stats = ethosu_delegateStats[ethosu_delegateCount++];
  stats.handle = handle;
  return &stats;
}
#endif
} // namespace

extern "C" {

#if defined(ET_ARM_ETHOSU_PROFILE_IO_COPIES)
void EthosUBackend_input_memcpy(size_t size) {
  ethosu_inputCopyStats.calls++;
  ethosu_inputCopyStats.bytes += size;
#if defined(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
  if (ethosu_activeDelegate != nullptr) {
    ethosu_activeDelegate->input_copies.calls++;
    ethosu_activeDelegate->input_copies.bytes += size;
  }
#endif
}

void EthosUBackend_output_memcpy(size_t size) {
  ethosu_outputCopyStats.calls++;
  ethosu_outputCopyStats.bytes += size;
#if defined(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
  if (ethosu_activeDelegate != nullptr) {
    ethosu_activeDelegate->output_copies.calls++;
    ethosu_activeDelegate->output_copies.bytes += size;
  }
#endif
}
#endif

#if defined(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
void EthosUBackend_delegate_begin(const void* handle) {
  ethosu_activeDelegate = get_delegate_stats(handle);
  if (ethosu_activeDelegate != nullptr) {
    ethosu_activeDelegate->backend_invocations++;
  }
}

void EthosUBackend_delegate_end() {
  ethosu_activeDelegate = nullptr;
}
#endif

void EthosUBackend_npu_begin() {
  ethosu_ArmWhenNPURunCycleCountStart = arm_pmu_cycles();
}

void EthosUBackend_npu_end() {
  ethosu_ArmWhenNPURunCycleCount +=
      arm_pmu_cycles() - ethosu_ArmWhenNPURunCycleCountStart;
  ++ethosu_delegation_count;
}

// Callback invoked at start of ArmBackend::execute()
void EthosUBackend_execute_begin() {
  // Save Cortex-M cycle clock to calculate total CPU cycles used in
  // ArmBackend_execute_end()
  ethosu_ArmBackendExecuteCycleCountStart = arm_pmu_cycles();
}

// Callback invoked at end of ArmBackend::execute()
void EthosUBackend_execute_end() {
  // Add Cortex-M cycle clock used during this ArmBackend::execute()
  ethosu_ArmBackendExecuteCycleCount +=
      (arm_pmu_cycles() - ethosu_ArmBackendExecuteCycleCountStart);
}
}

void ConsumeMeasurements() {
#if defined(ET_ARM_ETHOSU_PMU_STATS)
  for (size_t index = 0; index < ethosu_pmuBuffer.size; ++index) {
    const EthosUExecutionRecord& record = ethosu_pmuBuffer.records[index];
    ethosu_pmuCycleCount += record.stats.cycle_count;
#if defined(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
    if (auto* delegate = get_delegate_stats(
            reinterpret_cast<const void*>(record.delegate_handle))) {
      ++delegate->npu_invocations;
      delegate->pmu_cycles += record.stats.cycle_count;
    }
    DelegatePmuTotals* delegate_totals = nullptr;
    for (size_t entry = 0; entry < ethosu_delegatePmuCount; ++entry) {
      auto& candidate = ethosu_delegatePmuTotals[entry];
      if (candidate.handle == record.delegate_handle &&
          same_layout(candidate.totals.stats, record.stats)) {
        delegate_totals = &candidate;
        break;
      }
    }
    if (delegate_totals == nullptr &&
        ethosu_delegatePmuCount < ethosu_delegatePmuTotals.size()) {
      delegate_totals = &ethosu_delegatePmuTotals[ethosu_delegatePmuCount++];
      delegate_totals->handle = record.delegate_handle;
    }
    if (delegate_totals != nullptr) {
      accumulate_pmu_totals(delegate_totals->totals, record);
    } else {
      ethosu_delegatePmuCapacityExceeded = true;
    }
#endif
    DevicePmuTotals* totals = nullptr;
    for (size_t device = 0; device < ethosu_deviceCount; ++device) {
      if (same_layout(ethosu_deviceTotals[device].stats, record.stats)) {
        totals = &ethosu_deviceTotals[device];
        break;
      }
    }
    if (totals == nullptr && ethosu_deviceCount < ethosu_deviceTotals.size()) {
      totals = &ethosu_deviceTotals[ethosu_deviceCount++];
    }
    if (totals == nullptr) {
      ethosu_deviceCapacityExceeded = true;
      continue;
    }
    accumulate_pmu_totals(*totals, record);
  }
  ethosu_recordCapacityExceeded |= ethosu_pmuBuffer.reset();
#endif
}

void StartMeasurements() {
#if defined(__PMU_PRESENT) && (__PMU_PRESENT == 1U)
  // StopMeasurements() disables the cycle counter after each measurement.
  // Server mode starts a new measurement for every inference.
  ARM_PMU_Enable();
  ARM_PMU_CNTR_Enable(PMU_CNTENSET_CCNTR_ENABLE_Msk);
#endif
  ethosu_delegation_count = 0;
  ethosu_ArmBackendExecuteCycleCount = 0;
  ethosu_ArmWhenNPURunCycleCountStart = 0;
  ethosu_ArmWhenNPURunCycleCount = 0;
#if defined(ET_ARM_ETHOSU_PMU_STATS)
#if defined(ET_ARM_ETHOSU55_65_PMU_EVENTS)
  constexpr ethosu_pmu_event_type u55_u65_pmu_events[] = {
      ET_ARM_ETHOSU55_65_PMU_EVENTS};
  ET_CHECK_MSG(
      executorch::backends::arm::set_ethosu55_65_pmu_events(
          u55_u65_pmu_events) == executorch::runtime::Error::Ok,
      "Invalid Ethos-U55/U65 PMU event configuration");
#endif
#if defined(ET_ARM_ETHOSU85_PMU_EVENTS)
  constexpr ethosu_pmu_event_type u85_pmu_events[] = {
      ET_ARM_ETHOSU85_PMU_EVENTS};
  ET_CHECK_MSG(
      executorch::backends::arm::set_ethosu85_pmu_events(u85_pmu_events) ==
          executorch::runtime::Error::Ok,
      "Invalid Ethos-U85 PMU event configuration");
#endif
  ethosu_deviceTotals = {};
  ethosu_deviceCount = 0;
  ethosu_deviceCapacityExceeded = false;
  ethosu_recordCapacityExceeded = false;
  ethosu_pmuCycleCount = 0;
#if defined(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
  ethosu_delegatePmuTotals = {};
  ethosu_delegatePmuCount = 0;
  ethosu_delegatePmuCapacityExceeded = false;
#endif
  ethosu_pmuBuffer.reset();
  executorch::backends::arm::set_ethosu_stats_buffer(&ethosu_pmuBuffer);
#endif
#if defined(ET_ARM_ETHOSU_PROFILE_IO_COPIES)
  ethosu_inputCopyStats = {};
  ethosu_outputCopyStats = {};
#endif
#if defined(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
  ethosu_delegateStats = {};
  ethosu_delegateCount = 0;
  ethosu_activeDelegate = nullptr;
  ethosu_delegateCapacityExceeded = false;
#endif

  ethosu_ArmCycleCountStart = arm_pmu_cycles();
}

void StopMeasurements(size_t num_inferences) {
#if defined(__PMU_PRESENT) && (__PMU_PRESENT == 1U)
  ARM_PMU_CNTR_Disable(
      PMU_CNTENCLR_CCNTR_ENABLE_Msk | PMU_CNTENCLR_CNT0_ENABLE_Msk |
      PMU_CNTENCLR_CNT1_ENABLE_Msk);
#endif
  if (num_inferences == 0) {
    ET_LOG(Error, "Cannot report measurements without completed inferences");
    return;
  }
  if (!executorch::backends::arm::ethosu_execution_metrics_enabled()) {
    ET_LOG(
        Info,
        "WARNING: Ethos-U execution metrics are not linked; NPU delegations, "
        "wait cycles, and CPU/NPU ratios are incorrect");
  }
  uint64_t cycle_count = arm_pmu_cycles() - ethosu_ArmCycleCountStart;
  const bool cpu_timing_available =
#if defined(__PMU_PRESENT) && (__PMU_PRESENT == 1U)
      true;
#else
      false;
#endif
  if (!cpu_timing_available) {
    ET_LOG(Info, "CPU timing and CPU/NPU ratios: N/A (CPU PMU is not present)");
  }

  // Number of comand streams handled by the NPU
  ET_LOG(
      Info, "NPU Inferences : %lu", static_cast<printf_size_t>(num_inferences));
  ET_LOG(
      Info,
      "NPU delegations: %" PRIu32 " (%.2f per inference)",
      ethosu_delegation_count,
      (double)ethosu_delegation_count / num_inferences);
  if (cpu_timing_available) {
    ET_LOG(Info, "Profiler report, CPU cycles per operator:");
    ET_LOG(
        Info,
        "ethos-u : cycle_cnt : %" PRIu64 " cycles (%.2f per inference)",
        ethosu_ArmBackendExecuteCycleCount,
        (double)ethosu_ArmBackendExecuteCycleCount / num_inferences);
  }
#if defined(ET_ARM_ETHOSU_PROFILE_IO_COPIES)
  const uint64_t io_copy_calls =
      ethosu_inputCopyStats.calls + ethosu_outputCopyStats.calls;
  const uint64_t io_copy_bytes =
      ethosu_inputCopyStats.bytes + ethosu_outputCopyStats.bytes;
  ET_LOG(
      Info,
      "Ethos-U IO copy calls: %" PRIu64 " (%.2f per inference)",
      io_copy_calls,
      (double)io_copy_calls / num_inferences);
  ET_LOG(
      Info,
      "Ethos-U IO copy bytes: %" PRIu64 " bytes (%.2f per inference)",
      io_copy_bytes,
      (double)io_copy_bytes / num_inferences);
  ET_LOG(
      Info,
      "Ethos-U input copy: %" PRIu64 " calls, %" PRIu64 " bytes",
      ethosu_inputCopyStats.calls,
      ethosu_inputCopyStats.bytes);
  ET_LOG(
      Info,
      "Ethos-U output copy: %" PRIu64 " calls, %" PRIu64 " bytes",
      ethosu_outputCopyStats.calls,
      ethosu_outputCopyStats.bytes);
#endif
  // Only the Ethos-U backend is included in the operator total.
  if (cpu_timing_available) {
    ET_LOG(
        Info,
        "Operator(s) total: %" PRIu64 " CPU cycles (%.2f per inference)",
        ethosu_ArmBackendExecuteCycleCount,
        (double)ethosu_ArmBackendExecuteCycleCount / num_inferences);
    ET_LOG(
        Info,
        "Inference runtime: %" PRIu64 " CPU cycles total (%.2f per inference)",
        cycle_count,
        (double)cycle_count / num_inferences);
    if (cycle_count == 0 || ethosu_ArmWhenNPURunCycleCount > cycle_count) {
      ET_LOG(Info, "Inference CPU ratio: N/A");
      ET_LOG(Info, "Inference NPU ratio: N/A");
    } else {
      ET_LOG(
          Info,
          "NOTE: CPU/NPU ratios use CPU-clock time inside NPU callbacks, not NPU utilization");
      ET_LOG(
          Info,
          "Inference CPU ratio: %.2f %%",
          100.0 * (cycle_count - ethosu_ArmWhenNPURunCycleCount) / cycle_count);
      ET_LOG(
          Info,
          "Inference NPU ratio: %.2f %%",
          100.0 * ethosu_ArmWhenNPURunCycleCount / cycle_count);
    }
    ET_LOG(
        Info,
        "cpu_wait_for_npu_cntr : %" PRIu64 " CPU cycles (%.2f per inference)",
        ethosu_ArmWhenNPURunCycleCount,
        (double)ethosu_ArmWhenNPURunCycleCount / num_inferences);
  }

#if defined(ET_ARM_ETHOSU_PMU_STATS)
  ET_LOG(Info, "Ethos-U PMU report:");
  for (size_t device = 0; device < ethosu_deviceCount; ++device) {
    const DevicePmuTotals& totals = ethosu_deviceTotals[device];
    const EthosUExecutionStats& stats = totals.stats;
    ET_LOG(
        Info,
        "Ethos-U NPU 0x%" PRIxPTR ": product %" PRIu32 ", invocations %" PRIu64
        ", failures %" PRIu64,
        stats.device_identifier,
        stats.product,
        totals.invocations,
        totals.failures);
    ET_LOG(
        Info,
        "Ethos-U NPU 0x%" PRIxPTR " PMU cycles: %" PRIu64 " (overflow: %u)",
        stats.device_identifier,
        stats.cycle_count,
        static_cast<unsigned>(stats.cycle_overflow));
    for (uint32_t slot = 0; slot < stats.counter_count; ++slot) {
      const auto& counter = stats.counters[slot];
      ET_LOG(
          Info,
          "Ethos-U NPU 0x%" PRIxPTR " PMU counter %" PRIu32 " event %" PRIu32
          ": %" PRIu64 " (overflow: %u)",
          stats.device_identifier,
          slot,
          counter.event_id,
          counter.value,
          static_cast<unsigned>(counter.overflow));
    }
  }
  ET_LOG(
      Info,
      "ethosu_pmu_cycle_cntr : %" PRIu64 " (%.2f per inference)",
      ethosu_pmuCycleCount,
      (double)ethosu_pmuCycleCount / num_inferences);
  if (ethosu_recordCapacityExceeded) {
    ET_LOG(Error, "Ethos-U PMU record buffer overflowed; samples were lost");
  }
  if (ethosu_deviceCapacityExceeded) {
    ET_LOG(Error, "Ethos-U PMU device/layout capacity exceeded");
  }
#endif
#if defined(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
  ET_LOG(Info, "Ethos-U per-delegate PMU report:");
  for (size_t delegate_id = 0; delegate_id < ethosu_delegateCount;
       ++delegate_id) {
    const DelegateStats& stats = ethosu_delegateStats[delegate_id];
    ET_LOG(
        Info,
        "Ethos-U delegate %lu: %" PRIu64 " backend invocations, %" PRIu64
        " NPU invocations",
        static_cast<printf_size_t>(delegate_id),
        stats.backend_invocations,
        stats.npu_invocations);
    ET_LOG(
        Info,
        "Ethos-U delegate %lu PMU cycles: %" PRIu64,
        static_cast<printf_size_t>(delegate_id),
        stats.pmu_cycles);
    for (size_t entry = 0; entry < ethosu_delegatePmuCount; ++entry) {
      const DelegatePmuTotals& pmu_totals = ethosu_delegatePmuTotals[entry];
      if (pmu_totals.handle != reinterpret_cast<uintptr_t>(stats.handle)) {
        continue;
      }
      const EthosUExecutionStats& pmu = pmu_totals.totals.stats;
      for (uint32_t slot = 0; slot < pmu.counter_count; ++slot) {
        const auto& counter = pmu.counters[slot];
        ET_LOG(
            Info,
            "Ethos-U delegate %lu NPU 0x%" PRIxPTR " PMU counter %" PRIu32
            " event %" PRIu32 ": %" PRIu64 " (overflow: %u)",
            static_cast<printf_size_t>(delegate_id),
            pmu.device_identifier,
            slot,
            counter.event_id,
            counter.value,
            static_cast<unsigned>(counter.overflow));
      }
    }
#if defined(ET_ARM_ETHOSU_PROFILE_IO_COPIES)
    ET_LOG(
        Info,
        "Ethos-U delegate %lu input copy: %" PRIu64 " calls, %" PRIu64 " bytes",
        static_cast<printf_size_t>(delegate_id),
        stats.input_copies.calls,
        stats.input_copies.bytes);
    ET_LOG(
        Info,
        "Ethos-U delegate %lu output copy: %" PRIu64 " calls, %" PRIu64
        " bytes",
        static_cast<printf_size_t>(delegate_id),
        stats.output_copies.calls,
        stats.output_copies.bytes);
#endif
  }
  if (ethosu_delegateCapacityExceeded) {
    ET_LOG(
        Error,
        "Ethos-U per-delegate profiling exceeded its capacity of %lu delegates",
        static_cast<printf_size_t>(ethosu_delegateStats.size()));
  }
  if (ethosu_delegatePmuCapacityExceeded) {
    ET_LOG(Error, "Ethos-U per-delegate PMU device/layout capacity exceeded");
  }
#endif
}

#else
// cppcheck-suppress unusedFunction
void StartMeasurements() {}

// cppcheck-suppress unusedFunction
void ConsumeMeasurements() {}

// cppcheck-suppress unusedFunction
void StopMeasurements(size_t num_inferences) {
  (void)num_inferences;
}

#endif
