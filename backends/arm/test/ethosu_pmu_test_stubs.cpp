/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ethosu_pmu_test_stubs.h"

#include <algorithm>
#include <iterator>

MockPmu& pmu(ethosu_driver* driver) {
  return *static_cast<MockPmu*>(driver->dev.config->config);
}

std::atomic<uint32_t> observer_begins{0};
std::atomic<uint32_t> observer_ends{0};

extern "C" {
// Match the driver's weak hooks so the test verifies archive override behavior.
void __attribute__((weak)) ethosu_inference_begin(ethosu_driver*, void*) {}
void __attribute__((weak)) ethosu_inference_end(ethosu_driver*, void*) {}

void EthosUBackend_npu_begin() {
  ++observer_begins;
}

void EthosUBackend_npu_end() {
  ++observer_ends;
}

void ETHOSU_PMU_Enable(ethosu_driver* driver) {
  pmu(driver).enabled = true;
}

void ETHOSU_PMU_Disable(ethosu_driver* driver) {
  pmu(driver).enabled = false;
}

uint32_t ETHOSU_PMU_Get_NumEventCountersForDrv(ethosu_driver* driver) {
  return driver->dev.caps.product == 2 ? 8 : 4;
}

void ETHOSU_PMU_Set_EVTYPER(
    ethosu_driver* driver,
    uint32_t slot,
    ethosu_pmu_event_type event) {
  pmu(driver).events[slot] = event;
}

ethosu_pmu_event_type ETHOSU_PMU_Get_EVTYPER(
    ethosu_driver* driver,
    uint32_t slot) {
  return pmu(driver).events[slot];
}

void ETHOSU_PMU_PMCCNTR_CFG_Set_Stop_Event(
    ethosu_driver*,
    ethosu_pmu_event_type) {}
void ETHOSU_PMU_PMCCNTR_CFG_Set_Start_Event(
    ethosu_driver*,
    ethosu_pmu_event_type) {}

void ETHOSU_PMU_CNTR_Enable(ethosu_driver* driver, uint32_t mask) {
  pmu(driver).enabled_mask |= mask;
}

void ETHOSU_PMU_CNTR_Disable(ethosu_driver* driver, uint32_t mask) {
  pmu(driver).enabled_mask &= ~mask;
}

void ETHOSU_PMU_CYCCNT_Reset(ethosu_driver* driver) {
  pmu(driver).cycles = 0;
}

void ETHOSU_PMU_EVCNTR_ALL_Reset(ethosu_driver* driver) {
  std::fill(std::begin(pmu(driver).counts), std::end(pmu(driver).counts), 0);
}

void ETHOSU_PMU_Set_CNTR_OVS(ethosu_driver* driver, uint32_t mask) {
  pmu(driver).overflow &= ~mask;
}

uint32_t ETHOSU_PMU_Get_CNTR_OVS(ethosu_driver* driver) {
  return pmu(driver).overflow;
}

uint64_t ETHOSU_PMU_Get_CCNTR(ethosu_driver* driver) {
  return pmu(driver).cycles;
}

uint32_t ETHOSU_PMU_Get_EVCNTR(ethosu_driver* driver, uint32_t slot) {
  return pmu(driver).counts[slot];
}
}
