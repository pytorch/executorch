/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>

#include <ethosu_driver.h>
#include <pmu_ethosu.h>

#include <executorch/backends/arm/runtime/EthosUExecutionStats_Internal.h>

using namespace executorch::backends::arm;

extern "C" {
void ethosu_inference_begin(struct ethosu_driver* drv, void* user_arg) {
  auto* invocation = static_cast<EthosUInvocationContext*>(user_arg);
  if (invocation == nullptr) {
    return;
  }

  auto events = ethosu_prepare_invocation(*invocation, drv->dev.caps.product);
  if (!events.empty()) {
    const uint32_t event_count = events.size();
    const uint32_t counter_count =
        std::min(event_count, ETHOSU_PMU_Get_NumEventCountersForDrv(drv));
    if (counter_count != 0) {
      invocation->pmu_enabled = true;
      ETHOSU_PMU_Enable(drv);
      ETHOSU_PMU_PMCCNTR_CFG_Set_Stop_Event(drv, ETHOSU_PMU_NPU_IDLE);
      ETHOSU_PMU_PMCCNTR_CFG_Set_Start_Event(drv, ETHOSU_PMU_NPU_ACTIVE);
      uint32_t event_mask = 0;
      for (uint32_t slot = 0; slot < counter_count; ++slot) {
        const auto event = events[slot];
        ETHOSU_PMU_Set_EVTYPER(drv, slot, event);
        if (event != ETHOSU_PMU_NO_EVENT) {
          event_mask |= 1u << slot;
        }
      }
      ETHOSU_PMU_CNTR_Disable(drv, (1u << counter_count) - 1u);
      ETHOSU_PMU_CNTR_Enable(drv, event_mask | ETHOSU_PMU_CCNT_Msk);
      ETHOSU_PMU_CYCCNT_Reset(drv);
      ETHOSU_PMU_EVCNTR_ALL_Reset(drv);
      ETHOSU_PMU_Set_CNTR_OVS(drv, event_mask | ETHOSU_PMU_CCNT_Msk);
    }
  }

  ethosu_execution_begin(*invocation);
}

void ethosu_inference_end(struct ethosu_driver* drv, void* user_arg) {
  const auto* invocation = static_cast<EthosUInvocationContext*>(user_arg);
  if (invocation == nullptr) {
    return;
  }

  if (invocation->pmu_enabled) {
    const uint32_t counter_count = std::min<uint32_t>(
        drv->dev.caps.product == 2 ? 8u : 4u,
        ETHOSU_PMU_Get_NumEventCountersForDrv(drv));
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
    for (uint32_t slot = 0; slot < counter_count; ++slot) {
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
    ethosu_execution_end(*invocation, &record);
    return;
  }
  ethosu_execution_end(*invocation, nullptr);
}
}
