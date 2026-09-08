/* Copyright 2024-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <cinttypes>

#include "arm_perf_monitor.h"

#ifdef ETHOSU
#include <ethosu_driver.h>
#include <executorch/runtime/platform/log.h>
#include <pmu_ethosu.h>

namespace {

// Returns the Armv8.1-M PMU cycle counter; 0 on cores without it.
static inline uint64_t arm_pmu_cycles() {
#if defined(__PMU_PRESENT) && (__PMU_PRESENT == 1U)
  return ARM_PMU_Get_CCNTR();
#else
  return 0;
#endif
}

#if defined(ETHOSU55) || defined(ETHOSU65)
const uint32_t ethosu_pmuCountersUsed = 4;
#elif defined(ETHOSU85)
const uint32_t ethosu_pmuCountersUsed = 7;
#else
#error No NPU target defined
#endif

uint32_t ethosu_delegation_count = 0;
uint64_t ethosu_ArmCycleCountStart = 0;
uint64_t ethosu_ArmBackendExecuteCycleCountStart = 0;
uint64_t ethosu_ArmBackendExecuteCycleCount = 0;
uint64_t ethosu_ArmWhenNPURunCycleCountStart = 0;
uint64_t ethosu_ArmWhenNPURunCycleCount = 0;
uint64_t ethosu_pmuCycleCount = 0;
#if defined(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
struct DelegateStats {
  const void* handle = nullptr;
  uint64_t backend_invocations = 0;
  uint64_t npu_invocations = 0;
  uint64_t pmu_cycles = 0;
  std::array<uint64_t, ethosu_pmuCountersUsed> pmu_events{};
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
std::array<uint64_t, ethosu_pmuCountersUsed> ethosu_pmuEventCounts = {0};

// ethosu_pmuCountersUsed should match numbers of counters setup in
// ethosu_inference_begin() and not be more then the HW supports
static_assert(ETHOSU_PMU_NCOUNTERS >= ethosu_pmuCountersUsed);

} // namespace

extern "C" {

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

// Callback invoked at start of NPU execution
void ethosu_inference_begin(struct ethosu_driver* drv, void*) {
  // Enable PMU
  ETHOSU_PMU_Enable(drv);
  ETHOSU_PMU_PMCCNTR_CFG_Set_Stop_Event(drv, ETHOSU_PMU_NPU_IDLE);
  ETHOSU_PMU_PMCCNTR_CFG_Set_Start_Event(drv, ETHOSU_PMU_NPU_ACTIVE);

  // Setup 4 counters
#if defined(ETHOSU55) || defined(ETHOSU65)
  ETHOSU_PMU_Set_EVTYPER(drv, 0, ETHOSU_PMU_AXI0_RD_DATA_BEAT_RECEIVED);
  ETHOSU_PMU_Set_EVTYPER(drv, 1, ETHOSU_PMU_AXI1_RD_DATA_BEAT_RECEIVED);
  ETHOSU_PMU_Set_EVTYPER(drv, 2, ETHOSU_PMU_AXI0_WR_DATA_BEAT_WRITTEN);
  ETHOSU_PMU_Set_EVTYPER(drv, 3, ETHOSU_PMU_NPU_IDLE);
  // Enable the 4 counters
  ETHOSU_PMU_CNTR_Enable(
      drv,
      ETHOSU_PMU_CNT1_Msk | ETHOSU_PMU_CNT2_Msk | ETHOSU_PMU_CNT3_Msk |
          ETHOSU_PMU_CNT4_Msk);
#elif defined(ETHOSU85)
  ETHOSU_PMU_Set_EVTYPER(drv, 0, ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED);
  ETHOSU_PMU_Set_EVTYPER(drv, 1, ETHOSU_PMU_SRAM_WR_DATA_BEAT_WRITTEN);
  ETHOSU_PMU_Set_EVTYPER(drv, 2, ETHOSU_PMU_EXT_RD_DATA_BEAT_RECEIVED);
  ETHOSU_PMU_Set_EVTYPER(drv, 3, ETHOSU_PMU_EXT_WR_DATA_BEAT_WRITTEN);
  ETHOSU_PMU_Set_EVTYPER(drv, 4, ETHOSU_PMU_NPU_IDLE);
  ETHOSU_PMU_Set_EVTYPER(drv, 5, ETHOSU_PMU_MAC_ACTIVE);
  ETHOSU_PMU_Set_EVTYPER(drv, 6, ETHOSU_PMU_WD_ACTIVE);
  // Enable the 7 counters
  ETHOSU_PMU_CNTR_Enable(
      drv,
      ETHOSU_PMU_CNT1_Msk | ETHOSU_PMU_CNT2_Msk | ETHOSU_PMU_CNT3_Msk |
          ETHOSU_PMU_CNT4_Msk | ETHOSU_PMU_CNT5_Msk | ETHOSU_PMU_CNT6_Msk |
          ETHOSU_PMU_CNT7_Msk);
#else
#error No NPU target defined
#endif

  ETHOSU_PMU_CNTR_Enable(drv, ETHOSU_PMU_CCNT_Msk);
  ETHOSU_PMU_CYCCNT_Reset(drv);

  // Reset all counters
  ETHOSU_PMU_EVCNTR_ALL_Reset(drv);

  // Save Cortex-M cycle clock to calculate total CPU cycles used in
  // ethosu_inference_end()
  ethosu_ArmWhenNPURunCycleCountStart = arm_pmu_cycles();
}

// Callback invoked at end of NPU execution
void ethosu_inference_end(struct ethosu_driver* drv, void*) {
  ethosu_delegation_count++;
#if defined(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
  const uint64_t pmu_cycles = ETHOSU_PMU_Get_CCNTR(drv);
  ethosu_pmuCycleCount += pmu_cycles;
  if (ethosu_activeDelegate != nullptr) {
    ethosu_activeDelegate->npu_invocations++;
    ethosu_activeDelegate->pmu_cycles += pmu_cycles;
  }
#else
  ethosu_pmuCycleCount += ETHOSU_PMU_Get_CCNTR(drv);
#endif

  for (size_t i = 0; i < ethosu_pmuCountersUsed; i++) {
#if defined(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
    const uint64_t event_count = ETHOSU_PMU_Get_EVCNTR(drv, i);
    ethosu_pmuEventCounts[i] += event_count;
    if (ethosu_activeDelegate != nullptr) {
      ethosu_activeDelegate->pmu_events[i] += event_count;
    }
#else
    ethosu_pmuEventCounts[i] += ETHOSU_PMU_Get_EVCNTR(drv, i);
#endif
  }
  ETHOSU_PMU_Disable(drv);
  // Add Cortex-M cycle clock used during this NPU execution
  ethosu_ArmWhenNPURunCycleCount +=
      (arm_pmu_cycles() - ethosu_ArmWhenNPURunCycleCountStart);
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

void StartMeasurements() {
#if defined(__ARM_ARCH_8_1M_MAIN__)
  // StopMeasurements() disables the cycle counter after each measurement.
  // Server mode starts a new measurement for every inference.
  ARM_PMU_Enable();
  ARM_PMU_CNTR_Enable(PMU_CNTENSET_CCNTR_ENABLE_Msk);
#endif
  ethosu_delegation_count = 0;
  ethosu_ArmBackendExecuteCycleCount = 0;
  ethosu_ArmWhenNPURunCycleCount = 0;
  ethosu_pmuCycleCount = 0;
#if defined(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
  ethosu_delegateStats = {};
  ethosu_delegateCount = 0;
  ethosu_activeDelegate = nullptr;
  ethosu_delegateCapacityExceeded = false;
#endif

  for (size_t i = 0; i < ethosu_pmuCountersUsed; i++) {
    ethosu_pmuEventCounts[i] = 0;
  }
  ethosu_ArmCycleCountStart = arm_pmu_cycles();
}

void StopMeasurements(int num_inferences) {
#if defined(__PMU_PRESENT) && (__PMU_PRESENT == 1U)
  ARM_PMU_CNTR_Disable(
      PMU_CNTENCLR_CCNTR_ENABLE_Msk | PMU_CNTENCLR_CNT0_ENABLE_Msk |
      PMU_CNTENCLR_CNT1_ENABLE_Msk);
#endif
  uint64_t cycle_count = arm_pmu_cycles() - ethosu_ArmCycleCountStart;

  // Number of comand streams handled by the NPU
  ET_LOG(Info, "NPU Inferences : %d", num_inferences);
  ET_LOG(
      Info,
      "NPU delegations: %" PRIu32 " (%.2f per inference)",
      ethosu_delegation_count,
      (double)ethosu_delegation_count / num_inferences);
  ET_LOG(Info, "Profiler report, CPU cycles per operator:");
  // This is number of CPU cycles for the ethos-u operator from start to finish
  // in the framework If there is more then one commandstream the time is added
  // together
  ET_LOG(
      Info,
      "ethos-u : cycle_cnt : %" PRIu64 " cycles (%.2f per inference)",
      ethosu_ArmBackendExecuteCycleCount,
      (double)ethosu_ArmBackendExecuteCycleCount / num_inferences);
  // We could print a list of the cycles used by the other delegates here in the
  // future but now we only print ethos-u: this means that "Operator(s) total:
  // ..." will be the same number as ethos-u : cycle_cnt and not the sum of all
  ET_LOG(
      Info,
      "Operator(s) total: %" PRIu64 " CPU cycles (%.2f per inference)",
      ethosu_ArmBackendExecuteCycleCount,
      (double)ethosu_ArmBackendExecuteCycleCount / num_inferences);
  // Total CPU cycles used in the executorch method->execute()
  // Other delegates and no delegates are counted in this
  ET_LOG(
      Info,
      "Inference runtime: %" PRIu64 " CPU cycles total (%.2f per inference)",
      cycle_count,
      (double)cycle_count / num_inferences);

  ET_LOG(
      Info,
      "NOTE: CPU cycle values and ratio calculations require FPGA and identical CPU/NPU frequency");

  // Avoid division with zero if arm_pmu_cycles() is not enabled properly.
  if (cycle_count == 0) {
    ET_LOG(Info, "Inference CPU ratio: ?.?? %%");
    ET_LOG(Info, "Inference NPU ratio: ?.?? %%");
  } else {
    ET_LOG(
        Info,
        "Inference CPU ratio: %.2f %%",
        100.0 * (cycle_count - ethosu_ArmWhenNPURunCycleCount) / cycle_count);
    ET_LOG(
        Info,
        "Inference NPU ratio: %.2f %%",
        100.0 * ethosu_ArmWhenNPURunCycleCount / cycle_count);
  }

  // CPU cycles used by NPU, e.g. number of CPU cycles used between
  // ethosu_inference_begin() and ethosu_inference_end()
  // If there is more then one commandstream the time is added together
  ET_LOG(
      Info,
      "cpu_wait_for_npu_cntr : %" PRIu64 " CPU cycles (%.2f per inference)",
      ethosu_ArmWhenNPURunCycleCount,
      (double)ethosu_ArmWhenNPURunCycleCount / num_inferences);

  ET_LOG(Info, "Ethos-U PMU report:");
  ET_LOG(
      Info,
      "ethosu_pmu_cycle_cntr : % " PRIu64 " (%.2f per inference)",
      ethosu_pmuCycleCount,
      (double)ethosu_pmuCycleCount / num_inferences);

  for (size_t i = 0; i < ethosu_pmuCountersUsed; i++) {
    ET_LOG(
        Info,
        "ethosu_pmu_cntr%zd : %" PRIu64 " (%.2f per inference)",
        i,
        ethosu_pmuEventCounts[i],
        (double)ethosu_pmuEventCounts[i] / num_inferences);
  }
#if defined(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
  ET_LOG(Info, "Ethos-U per-delegate PMU report:");
  for (size_t delegate_id = 0; delegate_id < ethosu_delegateCount;
       ++delegate_id) {
    const DelegateStats& stats = ethosu_delegateStats[delegate_id];
    ET_LOG(
        Info,
        "Ethos-U delegate %zu: %" PRIu64 " backend invocations, %" PRIu64
        " NPU invocations",
        delegate_id,
        stats.backend_invocations,
        stats.npu_invocations);
    ET_LOG(
        Info,
        "Ethos-U delegate %zu PMU cycles: %" PRIu64,
        delegate_id,
        stats.pmu_cycles);
    for (size_t event = 0; event < ethosu_pmuCountersUsed; ++event) {
      ET_LOG(
          Info,
          "Ethos-U delegate %zu PMU counter %zu: %" PRIu64,
          delegate_id,
          event,
          stats.pmu_events[event]);
    }
  }
  if (ethosu_delegateCapacityExceeded) {
    ET_LOG(
        Error,
        "Ethos-U per-delegate profiling exceeded its capacity of %zu delegates",
        ethosu_delegateStats.size());
  }
#endif
#if defined(ETHOSU55) || defined(ETHOSU65)
  ET_LOG(
      Info,
      "Ethos-U PMU Events:[ETHOSU_PMU_AXI0_RD_DATA_BEAT_RECEIVED, ETHOSU_PMU_AXI1_RD_DATA_BEAT_RECEIVED, ETHOSU_PMU_AXI0_WR_DATA_BEAT_WRITTEN, ETHOSU_PMU_NPU_IDLE]");
#elif defined(ETHOSU85)
  ET_LOG(
      Info,
      "Ethos-U PMU Events:[ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED, ETHOSU_PMU_SRAM_WR_DATA_BEAT_WRITTEN, ETHOSU_PMU_EXT_RD_DATA_BEAT_RECEIVED, ETHOSU_PMU_EXT_WR_DATA_BEAT_WRITTEN, ETHOSU_PMU_NPU_IDLE, ETHOSU_PMU_MAC_ACTIVE, ETHOSU_PMU_WD_ACTIVE]");
#else
#error No NPU target defined
#endif
}

#else
// cppcheck-suppress unusedFunction
void StartMeasurements() {}

// cppcheck-suppress unusedFunction
void StopMeasurements(int num_inferences) {
  (void)num_inferences;
}

#endif
