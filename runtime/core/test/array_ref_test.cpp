/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/array_ref.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::ArrayRef;

TEST(TestArrayRef, ImplicitTypeConversion) {
  ArrayRef<int64_t> oneElement_1 = {1};
  EXPECT_EQ(oneElement_1.size(), 1);

  ArrayRef<int64_t> oneElement_2 = 1;
  EXPECT_EQ(oneElement_2.size(), 1);
}
