/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdio.h>
#include <stdlib.h>

#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>

#if defined(ESP_PLATFORM)
#include <esp_cpu.h>
#include <esp_heap_caps.h>
#include <esp_system.h>
#include <esp_clk_tree.h>
#endif

extern "C" {

void et_pal_init(void) {
#if defined(ESP_PLATFORM)
  ET_LOG(
      Info,
      "ESP32 ExecuTorch runner initialized. Free heap: %lu bytes.",
      static_cast<unsigned long>(esp_get_free_heap_size()));
#if defined(CONFIG_SPIRAM)
  ET_LOG(
      Info,
      "PSRAM available. Free PSRAM: %lu bytes.",
      static_cast<unsigned long>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)));
#endif
#endif
}

ET_NORETURN void et_pal_abort(void) {
#if defined(ESP_PLATFORM)
  esp_restart();
#else
  abort();
#endif
  while (1) {
  }
}

et_timestamp_t et_pal_current_ticks(void) {
#if defined(ESP_PLATFORM)
  return (et_timestamp_t)esp_cpu_get_cycle_count();
#else
  return 0;
#endif
}

et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) {
#if defined(ESP_PLATFORM)
  uint32_t cpu_freq_hz;
  if (esp_clk_tree_src_get_freq_hz(SOC_MOD_CLK_CPU, ESP_CLK_TREE_SRC_FREQ_PRECISION_CACHED, &cpu_freq_hz) ==
      ESP_OK) {
    return {1000000000u, cpu_freq_hz};
  }
#endif
  return {1000, 240}; // Default to 240 MHz if we can't get the actual frequency
}

void et_pal_emit_log_message(
    ET_UNUSED et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  printf(
      "%c [executorch:%s:%lu %s()] %s\n",
      level,
      filename,
      static_cast<unsigned long>(line),
      function,
      message);
  fflush(stdout);
}

void* et_pal_allocate(ET_UNUSED size_t size) {
  return nullptr;
}

void et_pal_free(ET_UNUSED void* ptr) {}

} // extern "C"