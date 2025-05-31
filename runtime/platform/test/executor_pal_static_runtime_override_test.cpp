/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/test/pal_spy.h>

namespace {
PalSpy spy = PalSpy();

void pal_init(void) {
  spy.init();
}

et_timestamp_t pal_current_ticks(void) {
  return spy.current_ticks();
}

et_tick_ratio_t pal_ticks_to_ns_multiplier(void) {
  return spy.ticks_to_ns_multiplier();
}

void pal_emit_log_message(
    et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    ET_UNUSED const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  spy.emit_log_message(
      timestamp, level, filename, function, line, message, length);
}

void* pal_allocate(size_t size) {
  return spy.allocate(size);
}

void pal_free(void* ptr) {
  spy.free(ptr);
}

// Statically register PAL impleementation.
bool registration_result =
    executorch::runtime::register_pal(executorch::runtime::PalImpl::create(
        pal_init,
        nullptr, // abort
        pal_current_ticks,
        pal_ticks_to_ns_multiplier,
        pal_emit_log_message,
        pal_allocate,
        pal_free,
        __FILE__));
} // namespace

TEST(RuntimePalOverrideTest, SmokeTest) {
  EXPECT_EQ(spy.current_ticks_call_count, 0);
  EXPECT_EQ(spy.allocate_call_count, 0);
  EXPECT_EQ(spy.free_call_count, 0);

  // Expect registration to call init.
  EXPECT_EQ(spy.init_call_count, 1);

  EXPECT_EQ(executorch::runtime::pal_current_ticks(), 1234);
  EXPECT_EQ(spy.current_ticks_call_count, 1);

  et_tick_ratio_t ticks_to_ns_multiplier =
      executorch::runtime::pal_ticks_to_ns_multiplier();
  EXPECT_EQ(ticks_to_ns_multiplier.numerator, 1);
  EXPECT_EQ(ticks_to_ns_multiplier.denominator, 1);

  executorch::runtime::pal_emit_log_message(
      5, kError, "test.cpp", "test_function", 6, "test message", 7);
  EXPECT_EQ(spy.emit_log_message_call_count, 1);
  EXPECT_EQ(spy.last_log_message_args.timestamp, 5);
  EXPECT_EQ(spy.last_log_message_args.level, kError);
  EXPECT_EQ(spy.last_log_message_args.filename, "test.cpp");
  EXPECT_EQ(spy.last_log_message_args.function, "test_function");
  EXPECT_EQ(spy.last_log_message_args.line, 6);
  EXPECT_EQ(spy.last_log_message_args.message, "test message");
  EXPECT_EQ(spy.last_log_message_args.length, 7);

  executorch::runtime::pal_allocate(16);
  EXPECT_EQ(spy.allocate_call_count, 1);

  executorch::runtime::pal_free(nullptr);
  EXPECT_EQ(spy.free_call_count, 1);
}
