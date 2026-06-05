/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * Performance monitoring helpers for Espressif ESP32/ESP32-S3.
 *
 * Uses the Xtensa/RISC-V CPU cycle counter (CCOUNT register on Xtensa,
 * or esp_cpu_get_cycle_count() from ESP-IDF) for timing measurements.
 */

void StartMeasurements();
void StopMeasurements(int num_inferences);
