/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/platform/test/stub_platform.h>

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>

#include <executorch/runtime/platform/compiler.h>
#include <executorch/runtime/platform/platform.h>

/**
 * @file
 * An implementation of the PAL to help with testing.
 */

static PlatformIntercept* platform_intercept;

void InterceptWith::install(PlatformIntercept* pi) {
  platform_intercept = pi;
}

/// Prints a message and aborts if an intercept is not installed.
#define ASSERT_INTERCEPT_INSTALLED()                                 \
  ({                                                                 \
    if (platform_intercept == nullptr) {                             \
      fprintf(stderr, "%s call was not intercepted\n", ET_FUNCTION); \
      fflush(stderr);                                                \
      __builtin_trap();                                              \
    }                                                                \
  })

extern "C" {

void et_pal_init(void) {
  ASSERT_INTERCEPT_INSTALLED();
  platform_intercept->init();
}

ET_NORETURN void et_pal_abort(void) {
  ASSERT_INTERCEPT_INSTALLED();
  // We can't properly intercept this since it's marked NORETURN.
  fprintf(stderr, "StubPlatform et_pal_abort called\n");
  fflush(stderr);
  __builtin_trap();
}

et_timestamp_t et_pal_current_ticks(void) {
  ASSERT_INTERCEPT_INSTALLED();
  return platform_intercept->current_ticks();
}

et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) {
  ASSERT_INTERCEPT_INSTALLED();
  return platform_intercept->ticks_to_ns_multiplier();
}

void et_pal_emit_log_message(
    et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    ET_UNUSED const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  ASSERT_INTERCEPT_INSTALLED();
  platform_intercept->emit_log_message(
      timestamp, level, filename, function, line, message, length);
}

void* et_pal_allocate(size_t size) {
  ASSERT_INTERCEPT_INSTALLED();
  return platform_intercept->allocate(size);
}

void et_pal_free(void* ptr) {
  ASSERT_INTERCEPT_INSTALLED();
  platform_intercept->free(ptr);
}

} // extern "C"

#include <gtest/gtest.h>

// Use a version of main() that does not call runtime_init().
//
// By default, executorch tests are built with a main() that calls
// runtime_init(), and ultimately et_pal_init(). The StubPlatform override of
// et_pal_init() will fail if it isn't intercepted, so we can't call it at start
// time.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // Purposefully do not initialize the PAL.
  return RUN_ALL_TESTS();
}
