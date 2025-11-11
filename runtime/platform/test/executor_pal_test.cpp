/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/runtime/platform/platform.h>

TEST(ExecutorPalTest, Initialization) {
  /*
   * Ensure `et_pal_init` can be called multiple times.
   * It has already been called once in the main() function.
   */
  et_pal_init();
}

TEST(ExecutorPalTest, TimestampCoherency) {
  et_pal_init();

  et_timestamp_t time_a = et_pal_current_ticks();
  ASSERT_TRUE(time_a >= 0);

  et_timestamp_t time_b = et_pal_current_ticks();
  ASSERT_TRUE(time_b >= time_a);
}

TEST(ExecutorPalTest, TickRateRatioSanity) {
  auto tick_ns_ratio = et_pal_ticks_to_ns_multiplier();
  ASSERT_TRUE(tick_ns_ratio.numerator > 0);
  ASSERT_TRUE(tick_ns_ratio.denominator > 0);
}
