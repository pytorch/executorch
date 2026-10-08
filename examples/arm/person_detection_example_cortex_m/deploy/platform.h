/* Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cstddef>
#include <cstdint>

struct PlatformModel {
  const uint8_t* data;
  size_t size;
};

void platform_init();
PlatformModel platform_model();
uint32_t platform_cycle_count();
uint64_t platform_cycles_to_microseconds(uint32_t cycles);
void platform_putc(uint8_t value);
void platform_flush();

extern "C" void person_detection_main();
