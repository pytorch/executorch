/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/platform/platform.h>
#include <cstdlib>

namespace executorch {
namespace runtime {

/**
 * The singleton instance of the PAL table.
 */
static pal_table global_pal_table = {
    .init = et_pal_init,
    .abort = et_pal_abort,
    .current_ticks = et_pal_current_ticks,
    .ticks_to_ns_multiplier = et_pal_ticks_to_ns_multiplier,
    .emit_log_message = et_pal_emit_log_message,
    .allocate = et_pal_allocate,
    .free = et_pal_free,
};

/**
 * Retrieve a pointer to the singleton instance of the PAL function table. This
 * can be used to override the default implementations of the PAL functions.
 */
pal_table* get_pal_table() {
  return &global_pal_table;
}

void pal_init() {
  get_pal_table()->init();
}

ET_NORETURN void pal_abort() {
  get_pal_table()->abort();
  // This should be unreachable, but in case the PAL implementation doesn't
  // abort, force it here.
  std::abort();
}

et_timestamp_t pal_current_ticks() {
  return get_pal_table()->current_ticks();
}

et_tick_ratio_t pal_ticks_to_ns_multiplier() {
  return get_pal_table()->ticks_to_ns_multiplier();
}

void pal_emit_log_message(
    et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    const char* function,
    size_t line,
    const char* message,
    size_t length) {
  get_pal_table()->emit_log_message(
      timestamp, level, filename, function, line, message, length);
}

void* pal_allocate(size_t size) {
  return get_pal_table()->allocate(size);
}

void pal_free(void* ptr) {
  get_pal_table()->free(ptr);
}

} // namespace runtime
} // namespace executorch
