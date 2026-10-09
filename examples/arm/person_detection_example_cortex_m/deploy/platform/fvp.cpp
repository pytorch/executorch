/* Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "platform.h"

#include "ARMCM55.h"
#include "system_ARMCM55.h"
#include "uart_stdout.h"

#if !defined(PERSON_DETECTION_MODEL_ADDR) || \
    !defined(PERSON_DETECTION_MODEL_SIZE)
#error "The FVP model location must be supplied by the application build."
#endif

void platform_init() {
  UartStdOutInit();
  ARM_PMU_Enable();
  ARM_PMU_CYCCNT_Reset();
  ARM_PMU_CNTR_Enable(PMU_CNTENSET_CCNTR_ENABLE_Msk);
}

PlatformModel platform_model() {
  return {
      reinterpret_cast<const uint8_t*>(PERSON_DETECTION_MODEL_ADDR),
      PERSON_DETECTION_MODEL_SIZE};
}

uint32_t platform_cycle_count() {
  return ARM_PMU_Get_CCNTR();
}

uint64_t platform_cycles_to_microseconds(uint32_t cycles) {
  return (static_cast<uint64_t>(cycles) * 1000000U) / SystemCoreClock;
}

void platform_putc(uint8_t value) {
  UartPutc(value);
}

void platform_flush() {}

int main() {
  person_detection_main();
  return 0;
}
