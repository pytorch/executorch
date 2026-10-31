/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Shared ExecuTorch platform abstraction layer (PAL) for ESP-IDF.
 *
 * This is the single ESP-IDF PAL source for the Espressif examples: the
 * executor runner and camera_stream compile it directly. Do not fork it
 * per consumer; a firmware image can only have one strong PAL
 * implementation.
 *
 * On ESP-IDF, logs go through ESP_LOG under the "executorch" tag.
 * ET_ESP_LOG_LEVEL is an optional esp_log_level_t compile-time floor;
 * IDF runtime levels still apply. Ticks are esp_timer nanoseconds, so
 * they stay monotonic across DFS and light sleep.
 */

#include <executorch/runtime/platform/compiler.h>
#include <executorch/runtime/platform/platform.h>

#include <cstdio>
#include <cstdlib>

#if defined(ESP_PLATFORM)
#include <esp_log.h>
#include <esp_system.h>
#include <esp_timer.h>

#ifndef ET_ESP_LOG_LEVEL
#define ET_ESP_LOG_LEVEL ESP_LOG_INFO
#endif

namespace {
const char* kTag = "executorch";

esp_log_level_t to_esp_level(et_pal_log_level_t level) {
  switch (level) {
    case et_pal_log_level_t::kDebug:
      return ESP_LOG_DEBUG;
    case et_pal_log_level_t::kInfo:
      return ESP_LOG_INFO;
    case et_pal_log_level_t::kError:
      return ESP_LOG_ERROR;
    case et_pal_log_level_t::kFatal:
      return ESP_LOG_ERROR;
    default:
      return ESP_LOG_INFO;
  }
}
} // namespace
#endif // defined(ESP_PLATFORM)

extern "C" {

void et_pal_init(void) {}

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
#if defined(ESP_PLATFORM)
  const esp_log_level_t esp_level = to_esp_level(level);
  if (esp_level > ET_ESP_LOG_LEVEL) {
    return;
  }
  // esp_log_write applies the runtime per-tag level as well. Keep the
  // runtime's file/function context: it is usually the only clue for
  // allocation and operator-mismatch failures.
  esp_log_write(
      esp_level, kTag, "%s:%zu %s: %s\n", filename, line, function, message);
#else
  printf(
      "%c [%s:%zu %s()] %s\n",
      static_cast<int>(level),
      filename,
      line,
      function,
      message);
  fflush(stdout);
#endif
}

void* et_pal_allocate(size_t size) {
  return malloc(size);
}

void et_pal_free(void* ptr) {
  free(ptr);
}

} // extern "C"
