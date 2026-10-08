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
#include <esp_heap_caps.h>
#include <esp_system.h>
#include <esp_timer.h>
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
}

et_timestamp_t et_pal_current_ticks(void) {
#if defined(ESP_PLATFORM)
  return static_cast<et_timestamp_t>(esp_timer_get_time()) * 1000;
#else
  return 0;
#endif
}

et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) {
  return {1, 1};
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

// cppcheck-suppress constParameterPointer
void et_pal_free(ET_UNUSED void* ptr) {}

} // extern "C"
