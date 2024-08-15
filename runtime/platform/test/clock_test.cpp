/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/platform/clock.h>

#include <executorch/runtime/platform/test/stub_platform.h>

#include <gtest/gtest.h>

using namespace ::testing;

class PalSpy : public PlatformIntercept {
 public:
  et_tick_ratio_t ticks_to_ns_multiplier() override {
    return tick_ns_multiplier;
  }

  et_tick_ratio_t tick_ns_multiplier = {1, 1};
};

TEST(ClockTest, ConvertTicksToNsSanity) {
  PalSpy spy;
  InterceptWith iw(spy);

  spy.tick_ns_multiplier = {3, 2};
  auto ns = executorch::runtime::ticks_to_ns(10);
  ASSERT_EQ(15, ns); // 10 ticks * 3/2 = 15 ns

  spy.tick_ns_multiplier = {2, 7};
  ns = executorch::runtime::ticks_to_ns(14);
  ASSERT_EQ(4, ns); // 14 ticks * 2/7 = 4 ns
}
