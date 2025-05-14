/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/function_ref.h>

#include <gtest/gtest.h>

using namespace ::testing;

using ::executorch::runtime::FunctionRef;

namespace {
void one(int32_t& i) {
  i = 1;
}

} // namespace

TEST(FunctionRefTest, CapturingLambda) {
  auto one = 1;
  auto f = [&](int32_t& i) { i = one; };
  int32_t val = 0;
  FunctionRef<void(int32_t&)>{f}(val);
  EXPECT_EQ(val, 1);
  // ERROR:
  // Item item1(0, f);
  // Item item2(0, [&](int32_t& i) { i = 2; });
  // FunctionRef<void(int32_t&)> ref([&](int32_t&){});
}

TEST(FunctionRefTest, NonCapturingLambda) {
  int32_t val = 0;
  FunctionRef<void(int32_t&)> ref([](int32_t& i) { i = 1; });
  ref(val);
  EXPECT_EQ(val, 1);

  val = 0;
  auto lambda = [](int32_t& i) { i = 1; };
  FunctionRef<void(int32_t&)> ref1(lambda);
  ref1(val);
  EXPECT_EQ(val, 1);
}

TEST(FunctionRefTest, FunctionPointer) {
  int32_t val = 0;
  FunctionRef<void(int32_t&)> ref(one);
  ref(val);
  EXPECT_EQ(val, 1);

  val = 0;
  FunctionRef<void(int32_t&)> ref2(one);
  ref2(val);
  EXPECT_EQ(val, 1);
}
