/*
 * Copyright 2026 Mwaniki Njagi.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cstdio>
#include <cstdlib>

#include <executorch/runtime/platform/platform.h>

#include <esp_heap_caps.h>
#include <esp_timer.h>

extern "C" {

void et_pal_init(void) {}

ET_NORETURN void et_pal_abort(void) {
  std::printf("ExecuTorch abort\n");
  std::abort();
}

et_timestamp_t et_pal_current_ticks(void) {
  return static_cast<et_timestamp_t>(esp_timer_get_time());
}

et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) {
  return {1000, 1};
}

void et_pal_emit_log_message(
    et_timestamp_t,
    et_pal_log_level_t level,
    const char* filename,
    const char*,
    size_t line,
    const char* message,
    size_t) {
  std::printf(
      "ET %c %s:%u %s\n",
      static_cast<char>(level),
      filename,
      static_cast<unsigned>(line),
      message);
}

void* et_pal_allocate(size_t size) {
  return heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
}

void et_pal_free(void* pointer) {
  heap_caps_free(pointer);
}
}
