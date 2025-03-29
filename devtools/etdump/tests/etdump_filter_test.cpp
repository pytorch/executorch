/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/devtools/etdump/etdump_filter.h>
#include <executorch/runtime/platform/runtime.h>

#include <cstring>

using ::executorch::etdump::ETDumpFilter;
using ::executorch::runtime::Error;
using ::executorch::runtime::kUnsetDebugHandle;
using ::executorch::runtime::Result;

class ETDumpFilterTest : public ::testing::Test {
 protected:
  ETDumpFilter filter;

  void SetUp() override {
    torch::executor::runtime_init();
  }

  void TearDown() override {}
};

TEST_F(ETDumpFilterTest, AddRegexPatternSuccess) {
  Result<bool> result = filter.add_regex("test.*");
  EXPECT_TRUE(result.ok());
  EXPECT_TRUE(result.get());
}

TEST_F(ETDumpFilterTest, AddRegexPatternFailure) {
  // Add maximum number of patterns
  for (size_t i = 0; i < ETDumpFilter::MAX_REGEX_PATTERNS; ++i) {
    Result<bool> result = filter.add_regex("pattern");
    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(result.get());
  }
  // Attempt to add one more pattern
  Result<bool> result = filter.add_regex("extra_pattern");
  EXPECT_FALSE(result.ok());
}

TEST_F(ETDumpFilterTest, SetDebugHandleRangeSuccess) {
  Result<bool> result = filter.set_debug_handle_range(10, 20);
  EXPECT_TRUE(result.ok());
  EXPECT_TRUE(result.get());
}

TEST_F(ETDumpFilterTest, SetDebugHandleRangeFailure) {
  Result<bool> result = filter.set_debug_handle_range(20, 10);
  EXPECT_EQ(result.error(), Error::InvalidArgument);
}

TEST_F(ETDumpFilterTest, FilterByNameSuccess) {
  filter.add_regex("event.*");
  Result<bool> result = filter.filter("event_name", kUnsetDebugHandle);
  EXPECT_TRUE(result.ok());
  EXPECT_TRUE(result.get());
}

TEST_F(ETDumpFilterTest, PartialMatchingFailed) {
  filter.add_regex("event.*");
  Result<bool> result = filter.filter("non_matching_event", kUnsetDebugHandle);
  EXPECT_TRUE(result.ok());
  EXPECT_FALSE(result.get());
}

TEST_F(ETDumpFilterTest, FilterByDelegateDebugIndexSuccess) {
  filter.set_debug_handle_range(10, 20);
  Result<bool> result = filter.filter(nullptr, 15);
  EXPECT_TRUE(result.ok());
  EXPECT_TRUE(result.get());
}

TEST_F(ETDumpFilterTest, FilterByDelegateDebugIndexFailure) {
  filter.set_debug_handle_range(10, 20);
  Result<bool> result = filter.filter(nullptr, 25);
  EXPECT_TRUE(result.ok());
  EXPECT_FALSE(result.get());
}

TEST_F(ETDumpFilterTest, NaiveFilterNameInputCanSucceed) {
  Result<bool> result = filter.filter("any_input", kUnsetDebugHandle);
  EXPECT_TRUE(result.ok());
  EXPECT_TRUE(result.get());
}

TEST_F(ETDumpFilterTest, NaiveFilterDebugHandleInputCanSucceed) {
  Result<bool> result = filter.filter(nullptr, 12345);
  EXPECT_TRUE(result.ok());
  EXPECT_TRUE(result.get());
}

TEST_F(ETDumpFilterTest, NoMatchFirstThenMatch) {
  filter.add_regex("non_matching_pattern");
  Result<bool> result_1 = filter.filter("matching_event", kUnsetDebugHandle);
  EXPECT_TRUE(result_1.ok());
  EXPECT_FALSE(result_1.get());
  filter.add_regex("matching_.*");
  Result<bool> result_2 = filter.filter("matching_event", kUnsetDebugHandle);
  EXPECT_TRUE(result_2.ok());
  EXPECT_TRUE(result_2.get());
}

TEST_F(ETDumpFilterTest, MatchRegexFirstThen) {
  filter.add_regex("matching.*");
  Result<bool> result_1 = filter.filter("matching_event", kUnsetDebugHandle);
  EXPECT_TRUE(result_1.ok());
  EXPECT_TRUE(result_1.get());
  filter.add_regex("non_matching_pattern");
  Result<bool> result_2 = filter.filter("matching_event", kUnsetDebugHandle);
  EXPECT_TRUE(result_2.ok());
  EXPECT_TRUE(result_2.get());
}
