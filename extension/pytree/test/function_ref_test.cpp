/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/pytree/function_ref.h>

#include <gtest/gtest.h>

using namespace ::testing;

using ::executorch::extension::pytree::FunctionRef;

namespace {
class Item {
 private:
  int32_t val_;
  FunctionRef<void(int32_t&)> ref_;

 public:
  /* implicit */ Item(int32_t val, FunctionRef<void(int32_t&)> ref)
      : val_(val), ref_(ref) {}

  int32_t get() {
    ref_(val_);
    return val_;
  }
};

void one(int32_t& i) {
  i = 1;
}

} // namespace

TEST(FunctionRefTest, CapturingLambda) {
  auto one = 1;
  auto f = [&](int32_t& i) { i = one; };
  Item item(0, FunctionRef<void(int32_t&)>{f});
  EXPECT_EQ(item.get(), 1);
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

  Item item(0, [](int32_t& i) { i = 1; });
  EXPECT_EQ(item.get(), 1);

  auto f = [](int32_t& i) { i = 1; };
  Item item1(0, f);
  EXPECT_EQ(item1.get(), 1);

  Item item2(0, std::move(f));
  EXPECT_EQ(item2.get(), 1);
}

TEST(FunctionRefTest, FunctionPointer) {
  int32_t val = 0;
  FunctionRef<void(int32_t&)> ref(one);
  ref(val);
  EXPECT_EQ(val, 1);

  Item item(0, one);
  EXPECT_EQ(item.get(), 1);

  Item item1(0, &one);
  EXPECT_EQ(item1.get(), 1);
}
