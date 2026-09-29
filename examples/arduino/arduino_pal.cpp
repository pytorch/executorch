/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * ExecuTorch platform layer written against the Arduino API rather than any
 * one board core. runtime/platform/default has a zephyr.cpp, but it calls
 * k_uptime_ticks and k_malloc, which pins the library to the single Arduino
 * core built on Zephyr. millis() and malloc() exist on all of them.
 *
 * The build script vendors this in place of the upstream backends and deletes
 * them; shipping two would leave the choice to link order.
 */

#include <Arduino.h>
#include <executorch/runtime/platform/compiler.h>
#include <executorch/runtime/platform/platform.h>
#include <cstdio>
#include <cstdlib>

/*
 * Logging goes through a weak hook so the library does not have to know how a
 * board exposes its console. On the Uno Q, Serial comes from
 * Arduino_RouterBridge rather than the core, and a sketch may not want the
 * runtime writing to it at all. Sketches that do implement this see the
 * runtime's own diagnostics -- allocation shortfalls, operator mismatches --
 * instead of a bare error code.
 */
extern "C" ET_WEAK void et_arduino_log(const char* message) {
  (void)message;
}

void et_pal_init(void) {}

ET_NORETURN void et_pal_abort(void) {
  // No _Exit on a microcontroller: there is nothing to return to. Park with
  // interrupts off so a watchdog resets the board rather than letting it run
  // on in an undefined state.
  noInterrupts();
  while (true) {
  }
}

et_timestamp_t et_pal_current_ticks(void) {
  return static_cast<et_timestamp_t>(micros());
}

et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) {
  // micros() ticks are microseconds, so nanoseconds are 1000/1 of them.
  return {1000, 1};
}

void et_pal_emit_log_message(
    ET_UNUSED et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    ET_UNUSED const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  char buffer[256];
  snprintf(
      buffer,
      sizeof(buffer),
      "%c [ET:%s:%zu] %s",
      static_cast<char>(level),
      filename,
      line,
      message);
  et_arduino_log(buffer);
}

void* et_pal_allocate(size_t size) {
  return malloc(size);
}

void et_pal_free(void* ptr) {
  free(ptr);
}
