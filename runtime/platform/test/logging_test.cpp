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

using namespace executorch::runtime;

class LoggingTest : public ::testing::Test {
 public:
  static void SetUpTestSuite() {
    // Initialize runtime.
    runtime_init();
  }
};

TEST_F(LoggingTest, LogLevels) {
  ET_LOG(Debug, "Debug log.");
  ET_LOG(Info, "Info log.");
  ET_LOG(Error, "Error log.");
  ET_LOG(Fatal, "Fatal log.");
}

TEST_F(LoggingTest, LogFormatting) {
  ET_LOG(Info, "Sample log with integer: %u", 100);
}
