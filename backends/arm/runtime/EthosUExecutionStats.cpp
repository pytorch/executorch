/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>

#include <ethosu_driver.h>
#include <pmu_ethosu.h>

#include <executorch/backends/arm/runtime/EthosUBackend_Pmu.h>
#include <executorch/backends/arm/runtime/EthosUExecutionStats.h>

namespace executorch::backends::arm {
namespace {
std::atomic<EthosUStatsBuffer*> capture_buffer{nullptr};
} // namespace

EthosUStatsBuffer* set_ethosu_stats_buffer(EthosUStatsBuffer* buffer) noexcept {
  return capture_buffer.exchange(buffer, std::memory_order_acq_rel);
}

bool ethosu_execution_metrics_enabled() noexcept {
  return true;
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
  if (invocation->buffer != nullptr) {
    static constexpr ethosu_pmu_event_type kU55U65Events[] = {
        ETHOSU_PMU_AXI0_RD_DATA_BEAT_RECEIVED,
        ETHOSU_PMU_AXI1_RD_DATA_BEAT_RECEIVED,
        ETHOSU_PMU_AXI0_WR_DATA_BEAT_WRITTEN,
        ETHOSU_PMU_NPU_IDLE,
    };
    static constexpr ethosu_pmu_event_type kU85Events[] = {
        ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED,
        ETHOSU_PMU_SRAM_WR_DATA_BEAT_WRITTEN,
        ETHOSU_PMU_EXT_RD_DATA_BEAT_RECEIVED,
        ETHOSU_PMU_EXT_WR_DATA_BEAT_WRITTEN,
        ETHOSU_PMU_NPU_IDLE,
        ETHOSU_PMU_MAC_ACTIVE,
        ETHOSU_PMU_WD_ACTIVE,
    };

    const ethosu_pmu_event_type* events = nullptr;
    uint32_t event_count = 0;
    if (drv->dev.caps.product == ETHOSU_PRODUCT_U55 ||
        drv->dev.caps.product == ETHOSU_PRODUCT_U65) {
      events = kU55U65Events;
      event_count = sizeof(kU55U65Events) / sizeof(kU55U65Events[0]);
    } else if (drv->dev.caps.product == ETHOSU_PRODUCT_U85) {
      events = kU85Events;
      event_count = sizeof(kU85Events) / sizeof(kU85Events[0]);
    }

    if (events != nullptr) {
      invocation->counter_count =
          std::min(event_count, ETHOSU_PMU_Get_NumEventCountersForDrv(drv));
    }
    if (events != nullptr && invocation->counter_count != 0) {
      ETHOSU_PMU_Enable(drv);
      ETHOSU_PMU_PMCCNTR_CFG_Set_Stop_Event(drv, ETHOSU_PMU_NPU_IDLE);
      ETHOSU_PMU_PMCCNTR_CFG_Set_Start_Event(drv, ETHOSU_PMU_NPU_ACTIVE);
      for (uint32_t slot = 0; slot < invocation->counter_count; ++slot) {
        ETHOSU_PMU_Set_EVTYPER(drv, slot, events[slot]);
      }
      const uint32_t event_mask = (1u << invocation->counter_count) - 1u;
      ETHOSU_PMU_CNTR_Enable(drv, event_mask | ETHOSU_PMU_CCNT_Msk);
      ETHOSU_PMU_CYCCNT_Reset(drv);
      ETHOSU_PMU_EVCNTR_ALL_Reset(drv);
      ETHOSU_PMU_Set_CNTR_OVS(drv, event_mask | ETHOSU_PMU_CCNT_Msk);
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

  if (invocation->buffer != nullptr && invocation->counter_count != 0) {
    executorch::backends::arm::EthosUExecutionRecord record{};
    record.delegate_handle = invocation->delegate_handle;
    record.success = drv->job.result == ETHOSU_JOB_RESULT_OK;
    record.stats.device_identifier = reinterpret_cast<uintptr_t>(drv->dev.reg);
    record.stats.product = drv->dev.caps.product;
    record.stats.cycle_count = ETHOSU_PMU_Get_CCNTR(drv);
    record.stats.cycle_count_valid = true;
    const uint32_t overflow_status = ETHOSU_PMU_Get_CNTR_OVS(drv);
    record.stats.cycle_overflow = (overflow_status & ETHOSU_PMU_CCNT_Msk) != 0;
    record.stats.counter_count = invocation->counter_count;
    for (uint32_t slot = 0; slot < invocation->counter_count; ++slot) {
      record.stats.counters[slot] = {
          static_cast<uint32_t>(ETHOSU_PMU_Get_EVTYPER(drv, slot)),
          ETHOSU_PMU_Get_EVCNTR(drv, slot),
          (overflow_status & (1u << slot)) != 0};
    }
    ETHOSU_PMU_Disable(drv);
    EthosUBackend_npu_end();
    invocation->buffer->append(record);
    return;
  }
  EthosUBackend_npu_end();
}
}
