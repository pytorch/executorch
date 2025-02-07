/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * Fallback PAL implementations that do not depend on any assumptions about
 * capabililties of the system.
 */

// This cpp file will provide weak implementations of the symbols declared in
// Platform.h. Client users can strongly define any or all of the functions to
// override them.
#define ET_INTERNAL_PLATFORM_WEAKNESS ET_WEAK
#include <executorch/runtime/platform/platform.h>

#include <executorch/runtime/platform/compiler.h>

void et_pal_init(void) {}

ET_NORETURN void et_pal_abort(void) {
  __builtin_trap();
}

et_timestamp_t et_pal_current_ticks(void) {
  // This file cannot make any assumptions about the presence of functions that
  // return the current time, so all users should provide a strong override for
  // it. To help make it more obvious when this weak version is being used,
  // return a number that should be easier to search for than 0.
  return 11223344;
}

et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) {
  // Since we don't define a tick rate, return a conversion ratio of 1.
  return {1, 1};
}

void et_pal_emit_log_message(
    ET_UNUSED et_timestamp_t timestamp,
    ET_UNUSED et_pal_log_level_t level,
    ET_UNUSED const char* filename,
    ET_UNUSED const char* function,
    ET_UNUSED size_t line,
    ET_UNUSED const char* message,
    ET_UNUSED size_t length) {}

void* et_pal_allocate(ET_UNUSED size_t size) {
  return nullptr;
}

void et_pal_free(ET_UNUSED void* ptr) {}
