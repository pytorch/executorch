/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/test/utils/alignment.h>

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

using namespace ::testing;

// The wrappers use the matcher and the underlying helper function, so testing
// them gives full coverage.

TEST(AlignmentTest, ExpectWrapper) {
  EXPECT_ALIGNED(reinterpret_cast<void*>(0xfff1000), 0x1000);
  EXPECT_ALIGNED(reinterpret_cast<void*>(0xfff1000), 0x100);
  EXPECT_ALIGNED(reinterpret_cast<void*>(0xfff1000), 0x10);
  EXPECT_ALIGNED(reinterpret_cast<void*>(0xfff1000), 0x8);
  EXPECT_ALIGNED(reinterpret_cast<void*>(0xfff1000), 0x4);
  EXPECT_ALIGNED(reinterpret_cast<void*>(0xfff1000), 0x2);
  EXPECT_ALIGNED(reinterpret_cast<void*>(0xfff1000), 0x1);
  EXPECT_NONFATAL_FAILURE(
      EXPECT_ALIGNED(reinterpret_cast<void*>(0xfff1001), 0x10), "");
  EXPECT_NONFATAL_FAILURE(
      EXPECT_ALIGNED(reinterpret_cast<void*>(0xfffffff), 0x10), "");
}

TEST(AlignmentTest, AssertWrapper) {
  ASSERT_ALIGNED(reinterpret_cast<void*>(0xfff1000), 0x1000);
  ASSERT_ALIGNED(reinterpret_cast<void*>(0xfff1000), 0x100);
  ASSERT_ALIGNED(reinterpret_cast<void*>(0xfff1000), 0x10);
  ASSERT_ALIGNED(reinterpret_cast<void*>(0xfff1000), 0x8);
  ASSERT_ALIGNED(reinterpret_cast<void*>(0xfff1000), 0x4);
  ASSERT_ALIGNED(reinterpret_cast<void*>(0xfff1000), 0x2);
  ASSERT_ALIGNED(reinterpret_cast<void*>(0xfff1000), 0x1);
  EXPECT_FATAL_FAILURE(
      ASSERT_ALIGNED(reinterpret_cast<void*>(0xfff1001), 0x10), "");
  EXPECT_FATAL_FAILURE(
      ASSERT_ALIGNED(reinterpret_cast<void*>(0xfffffff), 0x10), "");
}
