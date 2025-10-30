/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/runtime/platform/test/pal_spy.h>
#include <executorch/runtime/platform/test/stub_platform.h>

using namespace executorch::runtime;

class LoggingTest : public ::testing::Test {};

TEST_F(LoggingTest, LogLevels) {
  PalSpy spy;
  InterceptWith iw(spy);

  ET_LOG(Debug, "Debug log.");
  EXPECT_EQ(spy.last_log_message_args.message, "Debug log.");

  ET_LOG(Info, "Info log.");
  EXPECT_EQ(spy.last_log_message_args.message, "Info log.");

  ET_LOG(Error, "Error log.");
  EXPECT_EQ(spy.last_log_message_args.message, "Error log.");

  ET_LOG(Fatal, "Fatal log.");
  EXPECT_EQ(spy.last_log_message_args.message, "Fatal log.");
}

TEST_F(LoggingTest, LogFormatting) {
  PalSpy spy;
  InterceptWith iw(spy);

  ET_LOG(Info, "Sample log with integer: %u", 100);
  EXPECT_EQ(spy.last_log_message_args.message, "Sample log with integer: 100");
}

static std::string get_prefix(std::size_t length, bool use_multibyte) {
  if (!use_multibyte) {
    return std::string(length, 'A');
  }
  std::ostringstream result;
  result << std::string(length % 4, 'A');
  std::size_t remaining = length - (length % 4);
  while (remaining > 0) {
    result << "\xF0\x9F\x91\x8D";
    remaining -= 4;
  }
  return result.str();
}

TEST_F(LoggingTest, Utf8Truncation) {
  PalSpy spy;
  InterceptWith iw(spy);

  const char euro[] = "\xE2\x82\xAC";
  const char thumbs_up[] = "\xF0\x9F\x91\x8D";
  const char e_acute[] = "\xC3\xA9";
  const char capital_a_tilde[] = "\xC3\x83";

  struct TruncCase {
    size_t prefix_length;
    const char* codepoint;
  };
  const TruncCase cases[] = {
      {253, euro},
      {252, thumbs_up},
      {254, e_acute},
      {254, capital_a_tilde},
  };
  for (bool use_multibyte_prefix : {false, true}) {
    for (const auto& c : cases) {
      const std::string prefix =
          get_prefix(c.prefix_length, use_multibyte_prefix);
      const std::string suffix = "_SHOULD_BE_CUT";
      ET_LOG(Info, "%s%s%s", prefix.c_str(), c.codepoint, suffix.c_str());
      EXPECT_EQ(spy.last_log_message_args.message, prefix);
      EXPECT_EQ(spy.last_log_message_args.length, prefix.size());
    }
  }
}
