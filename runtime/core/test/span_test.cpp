/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/span.h>
#include <span>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::Span;

TEST(SpanTest, Ctors) {
  int64_t x[2] = {1, 2};

  Span<int64_t> span_range = {x, x + 2};
  Span<int64_t> span_array = {x};

  EXPECT_EQ(span_range.size(), 2);
  EXPECT_EQ(span_array.size(), 2);

  EXPECT_EQ(span_range.begin(), x);
  EXPECT_EQ(span_range.end(), x + 2);

  EXPECT_EQ(span_array.begin(), x);
  EXPECT_EQ(span_array.end(), x + 2);
}

TEST(SpanTest, MutableElements) {
  int64_t x[2] = {1, 2};
  Span<int64_t> span = {x, 2};
  EXPECT_EQ(span.size(), 2);
  EXPECT_EQ(span[0], 1);
  span[0] = 2;
  EXPECT_EQ(span[0], 2);
}

TEST(SpanTest, Empty) {
  int64_t x[2] = {1, 2};
  Span<int64_t> span_full = {x, 2};
  Span<int64_t> span_empty = {x, (size_t)0};

  EXPECT_FALSE(span_full.empty());
  EXPECT_TRUE(span_empty.empty());
}

TEST(SpanTest, Data) {
  int64_t x[2] = {1, 2};
  Span<int64_t> span = {x, 2};
  EXPECT_EQ(span.data(), x);
}

TEST(SpanTest, TriviallyCopyable) {
  int64_t x[2] = {1, 2};
  Span<int64_t> span = {x, 2};
  Span<int64_t> span_copy = span;
  EXPECT_EQ(span.data(), span_copy.data());
  EXPECT_EQ(span.size(), span_copy.size());
  EXPECT_TRUE(std::is_trivially_copyable<Span<int64_t>>::value);
}
