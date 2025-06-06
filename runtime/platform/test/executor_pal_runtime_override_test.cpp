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
PalSpy* active_spy;

void pal_init(void) {
  active_spy->init();
}

et_timestamp_t pal_current_ticks(void) {
  return active_spy->current_ticks();
}

et_tick_ratio_t pal_ticks_to_ns_multiplier(void) {
  return active_spy->ticks_to_ns_multiplier();
}

void pal_emit_log_message(
    et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    ET_UNUSED const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  active_spy->emit_log_message(
      timestamp, level, filename, function, line, message, length);
}

void* pal_allocate(size_t size) {
  return active_spy->allocate(size);
}

void pal_free(void* ptr) {
  active_spy->free(ptr);
}
} // namespace

class RuntimePalOverrideTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Capture the current PAL implementation so that it can be restored
    // after the test.
    _original_pal_impl = *executorch::runtime::get_pal_impl();
  }

  void TearDown() override {
    // Restore the original PAL implementation.

    // This is a slightly hacky way to allow this test to exist alongside
    // the executor_pal_override_test, which provides a build-time override
    // for et_pal_init. This implementation asserts that an intercept exists.
    // Since register_pal calls init, we need to make sure that an intercept
    // is registered. It will be deregistered when it goes out of scope,
    // allowing the tests to run in any order.
    InterceptWith iw(_spy);
    auto success = executorch::runtime::register_pal(_original_pal_impl);
    if (!success) {
      throw std::runtime_error("Failed to restore PAL implementation.");
    }
  }

  void RegisterSpy() {
    active_spy = &_spy;

    executorch::runtime::register_pal(executorch::runtime::PalImpl::create(
        pal_init,
        nullptr, // abort
        pal_current_ticks,
        pal_ticks_to_ns_multiplier,
        pal_emit_log_message,
        pal_allocate,
        pal_free,
        __FILE__));
  }

  PalSpy _spy;

 private:
  // The PAL implementation at the time of setup.
  executorch::runtime::PalImpl _original_pal_impl;
};

TEST_F(RuntimePalOverrideTest, SmokeTest) {
  EXPECT_EQ(_spy.init_call_count, 0);
  EXPECT_EQ(_spy.current_ticks_call_count, 0);
  EXPECT_EQ(_spy.allocate_call_count, 0);
  EXPECT_EQ(_spy.free_call_count, 0);

  RegisterSpy();

  // Expect register to call init.
  EXPECT_EQ(_spy.init_call_count, 1);

  EXPECT_EQ(executorch::runtime::pal_current_ticks(), 1234);
  EXPECT_EQ(_spy.current_ticks_call_count, 1);

  et_tick_ratio_t ticks_to_ns_multiplier =
      executorch::runtime::pal_ticks_to_ns_multiplier();
  EXPECT_EQ(ticks_to_ns_multiplier.numerator, 1);
  EXPECT_EQ(ticks_to_ns_multiplier.denominator, 1);

  executorch::runtime::pal_emit_log_message(
      5, kError, "test.cpp", "test_function", 6, "test message", 7);
  EXPECT_EQ(_spy.emit_log_message_call_count, 1);
  EXPECT_EQ(_spy.last_log_message_args.timestamp, 5);
  EXPECT_EQ(_spy.last_log_message_args.level, kError);
  EXPECT_EQ(_spy.last_log_message_args.filename, "test.cpp");
  EXPECT_EQ(_spy.last_log_message_args.function, "test_function");
  EXPECT_EQ(_spy.last_log_message_args.line, 6);
  EXPECT_EQ(_spy.last_log_message_args.message, "test message");
  EXPECT_EQ(_spy.last_log_message_args.length, 7);

  executorch::runtime::pal_allocate(16);
  EXPECT_EQ(_spy.allocate_call_count, 1);

  executorch::runtime::pal_free(nullptr);
  EXPECT_EQ(_spy.free_call_count, 1);
}
