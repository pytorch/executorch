/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cinttypes>

#include "esp_perf_monitor.h"

#if defined(ESP_PLATFORM)

#include <esp_cpu.h>
#include <esp_system.h>
#include <esp_timer.h>
#include <executorch/runtime/platform/log.h>

namespace {

uint64_t start_cycle_count = 0;
int64_t start_time_us = 0;

} // namespace

void StartMeasurements() {
  start_cycle_count = esp_cpu_get_cycle_count();
  start_time_us = esp_timer_get_time();
}

void StopMeasurements(int num_inferences) {
  uint64_t end_cycle_count = esp_cpu_get_cycle_count();
  int64_t end_time_us = esp_timer_get_time();

  uint64_t total_cycles = end_cycle_count - start_cycle_count;
  int64_t total_time_us = end_time_us - start_time_us;

  ET_LOG(Info, "Profiler report:");
  ET_LOG(Info, "Number of inferences: %d", num_inferences);

  // Guard against division by zero or invalid counts when computing
  // per-inference metrics.
  if (num_inferences <= 0) {
    ET_LOG(
        Info,
        "Total CPU cycles: %" PRIu64 " (per-inference metrics not computed)",
        total_cycles);
    ET_LOG(
        Info,
        "Total wall time: %" PRId64 " us (per-inference metrics not computed)",
        total_time_us);
    // Log ESP32 system memory info
    ET_LOG(
        Info,
        "Free heap: %lu bytes",
        static_cast<unsigned long>(esp_get_free_heap_size()));
    ET_LOG(
        Info,
        "Min free heap ever: %lu bytes",
        static_cast<unsigned long>(esp_get_minimum_free_heap_size()));
    return;
  }

  ET_LOG(
      Info,
      "Total CPU cycles: %" PRIu64 " (%.2f per inference)",
      total_cycles,
      (double)total_cycles / num_inferences);
  ET_LOG(
      Info,
      "Total wall time: %" PRId64 " us (%.2f us per inference)",
      total_time_us,
      (double)total_time_us / num_inferences);
  ET_LOG(
      Info,
      "Average inference time: %.3f ms",
      (double)total_time_us / num_inferences / 1000.0);

  // Log ESP32 system memory info
  ET_LOG(
      Info,
      "Free heap: %lu bytes",
      static_cast<unsigned long>(esp_get_free_heap_size()));
  ET_LOG(
      Info,
      "Min free heap ever: %lu bytes",
      static_cast<unsigned long>(esp_get_minimum_free_heap_size()));
}

#else // !defined(ESP_PLATFORM)

// Stub implementation for non-ESP builds (e.g. host testing)
void StartMeasurements() {}

void StopMeasurements(int num_inferences) {
  (void)num_inferences;
}

#endif // defined(ESP_PLATFORM)
