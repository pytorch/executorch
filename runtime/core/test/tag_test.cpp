/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/tag.h>

#include <gtest/gtest.h>
#include <array>

using namespace ::testing;
using executorch::runtime::kTagNameBufferSize;
using executorch::runtime::Tag;
using executorch::runtime::tag_to_string;

// The behavior of tag_to_string depends on the value of ET_ENABLE_ENUM_STRINGS.
// If it is not set, tag_to_string will return a string representation of the
// enum index value. As this behavior is compile-time gated, tests must also
// be compile-time gated.
#if ET_ENABLE_ENUM_STRINGS
TEST(TagToString, TagValues) {
  std::array<char, 16> name;

  tag_to_string(Tag::Tensor, name.data(), name.size());
  EXPECT_STREQ("Tensor", name.data());

  tag_to_string(Tag::Int, name.data(), name.size());
  EXPECT_STREQ("Int", name.data());

  tag_to_string(Tag::Double, name.data(), name.size());
  EXPECT_STREQ("Double", name.data());

  tag_to_string(Tag::Bool, name.data(), name.size());
  EXPECT_STREQ("Bool", name.data());
}

TEST(TagToString, TagNameBufferSize) {
  // Validate that kTagNameBufferSize is large enough to hold the all tag
  // strings without truncation.
  std::array<char, kTagNameBufferSize> name;

  // Note that the return value of tag_to_string does not include the null
  // terminator.
  size_t longest = 0;

#define TEST_CASE(tag)                                                \
  auto tag##_len = tag_to_string(Tag::tag, name.data(), name.size()); \
  EXPECT_LT(tag##_len, kTagNameBufferSize)                            \
      << "kTagNameBufferSize is too small to hold " #tag;             \
  longest = std::max(longest, tag##_len);

  EXECUTORCH_FORALL_TAGS(TEST_CASE)
#undef TEST_CASE

  EXPECT_EQ(longest + 1, kTagNameBufferSize)
      << "kTagNameBufferSize has incorrect value, expected " << longest + 1;
}

TEST(TagToString, FitsExact) {
  std::array<char, 4> name;

  auto ret = tag_to_string(Tag::Int, name.data(), name.size());

  EXPECT_EQ(3, ret);
  EXPECT_STREQ("Int", name.data());
}

TEST(TagToString, Truncate) {
  std::array<char, 6> name;
  std::fill(name.begin(), name.end(), '-');

  auto ret = tag_to_string(Tag::Double, name.data(), name.size());
  EXPECT_EQ(6, ret);
  EXPECT_TRUE(name[name.size() - 1] == 0);
  EXPECT_STREQ("Doubl", name.data());
}
#endif // ET_ENABLE_ENUM_STRINGS
