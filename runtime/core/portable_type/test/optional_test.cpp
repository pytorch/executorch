/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/portable_type/optional.h>

#include <string>
#include <type_traits>

#include <executorch/test/utils/DeathTest.h>
#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::etensor::nullopt;
using executorch::runtime::etensor::optional;

// Test that optional::value_type matches the template parameter type.
static_assert(
    std::is_same<optional<int32_t>::value_type, int32_t>::value,
    "Unexpected optional::value_type");
static_assert(
    std::is_same<optional<std::string>::value_type, std::string>::value,
    "Unexpected optional::value_type");

TEST(TestOptional, DefaultHasNoValue) {
  optional<int32_t> o;
  EXPECT_FALSE(o.has_value());
}

TEST(TestOptional, NulloptHasNoValue) {
  optional<int32_t> o(nullopt);
  EXPECT_FALSE(o.has_value());
}

TEST(TestOptional, ValueOfEmptyOptionalShouldDie) {
  optional<int32_t> o;
  EXPECT_FALSE(o.has_value());

  ET_EXPECT_DEATH({ (void)o.value(); }, "");
}

TEST(TestOptional, IntValue) {
  optional<int32_t> o(15);
  EXPECT_TRUE(o.has_value());
  EXPECT_EQ(o.value(), 15);
}

TEST(TestOptional, NonTrivialValueType) {
  optional<std::string> o("hey");
  EXPECT_TRUE(o.has_value());
  EXPECT_EQ(o.value(), "hey");
}

TEST(TestOptional, ConstValue) {
  const optional<std::string> o("hey");
  auto s = o.value(); // If this compiles, we're good.
  EXPECT_EQ(o.value(), "hey");
}

TEST(TestOptional, CopyCtorWithValue) {
  optional<int32_t> o1(15);
  optional<int32_t> o2(o1);

  EXPECT_TRUE(o2.has_value());
  EXPECT_EQ(o2.value(), 15);
}

TEST(TestOptional, CopyCtorWithNoValue) {
  optional<int32_t> o1;
  optional<int32_t> o2(o1);

  EXPECT_FALSE(o2.has_value());
}

TEST(TestOptional, CopyAssignTrivial) {
  optional<int32_t> o1(1);
  optional<int32_t> o2(2);
  o1 = o2;

  EXPECT_EQ(o1.value(), 2);
}

TEST(TestOptional, CopyAssignNonTrivial) {
  optional<std::string> o1("abcde");
  optional<std::string> o2("foo");
  o1 = o2;

  EXPECT_EQ(o1.value(), "foo");
}

TEST(TestOptional, CopyAssignNone) {
  optional<int32_t> o1(2);
  optional<int32_t> o2;
  o1 = o2;
  EXPECT_FALSE(o1.has_value());
}

TEST(TestOptional, MoveCtorWithNoValue) {
  optional<int32_t> o1;
  optional<int32_t> o2(std::move(o1));

  EXPECT_FALSE(o2.has_value());
}

TEST(TestOptional, MoveCtorWithValue) {
  optional<int32_t> o1(15);
  optional<int32_t> o2(std::move(o1));

  EXPECT_TRUE(o2.has_value());
  EXPECT_EQ(o2.value(), 15);
}

TEST(TestOptional, MoveCtorNonTrivialType) {
  optional<std::string> o1("abc");
  optional<std::string> o2(std::move(o1));

  EXPECT_TRUE(o2.has_value());
  EXPECT_EQ(o2.value(), "abc");
}

optional<int32_t> function_returning_optional_of(int32_t value) {
  return value;
}

TEST(TestOptional, ImplicitReturnOfValue) {
  auto o = function_returning_optional_of(21);
  EXPECT_TRUE(o.has_value());
  EXPECT_EQ(o.value(), 21);
}

optional<int32_t> function_returning_nullopt() {
  return nullopt;
}

TEST(TestOptional, ImplicitReturnOfNullopt) {
  auto o = function_returning_nullopt();
  EXPECT_FALSE(o.has_value());
}
