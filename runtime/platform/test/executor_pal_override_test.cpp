/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/runtime/platform/test/stub_platform.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::LogLevel;

class PalSpy : public PlatformIntercept {
 public:
  PalSpy() = default;

  void init() override {
    ++init_call_count;
  }

  static constexpr et_timestamp_t kTimestamp = 1234;

  et_timestamp_t current_ticks() override {
    ++current_ticks_call_count;
    return kTimestamp;
  }

  et_tick_ratio_t ticks_to_ns_multiplier() override {
    return tick_ns_multiplier;
  }

  void emit_log_message(
      et_timestamp_t timestamp,
      et_pal_log_level_t level,
      const char* filename,
      const char* function,
      size_t line,
      const char* message,
      size_t length) override {
    ++emit_log_message_call_count;
    last_log_message_args.timestamp = timestamp;
    last_log_message_args.level = level;
    last_log_message_args.filename = filename;
    last_log_message_args.function = function;
    last_log_message_args.line = line;
    last_log_message_args.message = message;
    last_log_message_args.length = length;
  }

  virtual ~PalSpy() = default;

  size_t init_call_count = 0;
  size_t current_ticks_call_count = 0;
  size_t emit_log_message_call_count = 0;
  et_tick_ratio_t tick_ns_multiplier = {1, 1};

  /// The args that were passed to the most recent call to emit_log_message().
  struct {
    et_timestamp_t timestamp;
    et_pal_log_level_t level;
    std::string filename; // Copy of the char* to avoid lifetime issues.
    std::string function;
    size_t line;
    std::string message;
    size_t length;
  } last_log_message_args = {};
};

// Demonstrate what would happen if we didn't intercept the PAL calls.
TEST(ExecutorPalOverrideTest, DiesIfNotIntercepted) {
  ET_EXPECT_DEATH(
      executorch::runtime::runtime_init(),
      "et_pal_init call was not intercepted");
}

TEST(ExecutorPalOverrideTest, InitIsRegistered) {
  PalSpy spy;
  InterceptWith iw(spy);

  EXPECT_EQ(spy.init_call_count, 0);
  executorch::runtime::runtime_init();
  EXPECT_EQ(spy.init_call_count, 1);
}

#if ET_LOG_ENABLED
TEST(ExecutorPalOverrideTest, LogSmokeTest) {
  PalSpy spy;
  InterceptWith iw(spy);

  EXPECT_EQ(spy.current_ticks_call_count, 0);
  EXPECT_EQ(spy.emit_log_message_call_count, 0);

  // Use the highest log level, which isn't likely to be disabled.
  ASSERT_GE(LogLevel::Fatal, LogLevel::ET_MIN_LOG_LEVEL);
  ET_LOG(Fatal, "Test log");

  EXPECT_EQ(spy.emit_log_message_call_count, 1);
  // Logging a message should also cause et_pal_current_ticks to be called once.
  EXPECT_EQ(spy.current_ticks_call_count, 1);

  const auto& args = spy.last_log_message_args;
  EXPECT_EQ(args.timestamp, PalSpy::kTimestamp);
  EXPECT_EQ(args.level, et_pal_log_level_t::kFatal);
  // Ignore filename/function/line to avoid fragility.
  EXPECT_EQ(args.message, "Test log");
  EXPECT_EQ(args.length, sizeof("Test log") - 1);
}

TEST(ExecutorPalOverrideTest, LogLevels) {
  PalSpy spy;
  InterceptWith iw(spy);
  const auto& args = spy.last_log_message_args;

  // Test all log levels. Demonstrates the mapping between LogLevel and
  // et_pal_log_level_t.
  if (LogLevel::Debug >= LogLevel::ET_MIN_LOG_LEVEL) {
    ET_LOG(Debug, "Test log");
    EXPECT_EQ(args.level, et_pal_log_level_t::kDebug);
  }

  if (LogLevel::Info >= LogLevel::ET_MIN_LOG_LEVEL) {
    ET_LOG(Info, "Test log");
    EXPECT_EQ(args.level, et_pal_log_level_t::kInfo);
  }

  if (LogLevel::Error >= LogLevel::ET_MIN_LOG_LEVEL) {
    ET_LOG(Error, "Test log");
    EXPECT_EQ(args.level, et_pal_log_level_t::kError);
  }

  if (LogLevel::Fatal >= LogLevel::ET_MIN_LOG_LEVEL) {
    ET_LOG(Fatal, "Test log");
    EXPECT_EQ(args.level, et_pal_log_level_t::kFatal);
  }

  // An invalid LogLevel should map to kUnknown.
  ET_LOG(NumLevels, "Test log");
  EXPECT_EQ(args.level, et_pal_log_level_t::kUnknown);
}

TEST(ExecutorPalOverrideTest, TickToNsMultiplier) {
  PalSpy spy;
  InterceptWith iw(spy);

  // Validate that tick to ns multipliers are overridden.
  spy.tick_ns_multiplier = {2, 3};
  EXPECT_EQ(et_pal_ticks_to_ns_multiplier().numerator, 2);
  EXPECT_EQ(et_pal_ticks_to_ns_multiplier().denominator, 3);

  spy.tick_ns_multiplier = {3, 1};
  EXPECT_EQ(et_pal_ticks_to_ns_multiplier().numerator, 3);
  EXPECT_EQ(et_pal_ticks_to_ns_multiplier().denominator, 1);
}

#endif
