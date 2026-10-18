/*
 * Copyright 2026 Mwaniki Njagi.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cstdint>
#include <cstdio>

#include <esp_chip_info.h>
#include <esp_flash.h>
#include <esp_heap_caps.h>
#include <esp_idf_version.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <sdkconfig.h>

#include <runner.h>

extern "C" {
extern const uint8_t espnn_bundle_start[];
extern const uint8_t espnn_bundle_end[];
}

namespace {

void print_chip() {
  esp_chip_info_t chip{};
  esp_chip_info(&chip);
  uint32_t flash_bytes = 0;
  esp_flash_get_size(nullptr, &flash_bytes);
  std::printf(
      "CHIP target=%s cores=%d revision=%d cpu_mhz=%d flash_mb=%u psram_kb=%u "
      "internal_free_kb=%u idf=%s\n",
      CONFIG_IDF_TARGET,
      chip.cores,
      chip.revision,
      CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ,
      static_cast<unsigned>(flash_bytes / (1024 * 1024)),
      static_cast<unsigned>(heap_caps_get_total_size(MALLOC_CAP_SPIRAM) / 1024),
      static_cast<unsigned>(
          heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT) /
          1024),
      esp_get_idf_version());
}

void runner_task(void*) {
  print_chip();
  const size_t size = reinterpret_cast<uintptr_t>(espnn_bundle_end) -
      reinterpret_cast<uintptr_t>(espnn_bundle_start);
  bool pass = false;
  if (executorch::examples::espressif::is_int8_bundle(
          espnn_bundle_start, size)) {
    pass = executorch::examples::espressif::run_int8_suite(
        espnn_bundle_start, size);
  } else if (executorch::examples::espressif::is_pte_bundle(
                 espnn_bundle_start, size)) {
#if CONFIG_ESPNN_EXECUTORCH_DELEGATE
    pass = executorch::examples::espressif::run_pte_suite(
        espnn_bundle_start, size);
#else
    std::printf(
        "ERROR: this bundle holds .pte programs; enable CONFIG_ESPNN_EXECUTORCH_DELEGATE\n");
#endif
  } else {
    std::printf(
        "ERROR: unrecognised bundle; expected XTB1 (int8 graphs) or XTP1 (.pte programs)\n");
  }
  std::printf("ESPNN RUNNER: %s\n", pass ? "PASS" : "FAIL");
  while (true)
    vTaskDelay(pdMS_TO_TICKS(10000));
}

} // namespace

extern "C" void app_main() {
  xTaskCreatePinnedToCore(
      runner_task,
      "espnn_runner",
      CONFIG_ESPNN_RUNNER_STACK_KB * 1024,
      nullptr,
      5,
      nullptr,
      0);
}
