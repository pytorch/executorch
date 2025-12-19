/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/aoti/slim/util/type_convert.h>

namespace executorch::backends::aoti::slim {
namespace {

TEST(TypeConvertTest, ToInt32Vec) {
  std::vector<int64_t> int64_vec = {1, 2, 3, 4, 5};
  auto int32_vec = to_int32_vec(int64_vec);

  EXPECT_EQ(int32_vec.size(), 5);
  EXPECT_EQ(int32_vec[0], 1);
  EXPECT_EQ(int32_vec[1], 2);
  EXPECT_EQ(int32_vec[2], 3);
  EXPECT_EQ(int32_vec[3], 4);
  EXPECT_EQ(int32_vec[4], 5);
}

TEST(TypeConvertTest, ToInt64Vec) {
  std::vector<int32_t> int32_vec = {10, 20, 30};
  auto int64_vec = to_int64_vec(int32_vec);

  EXPECT_EQ(int64_vec.size(), 3);
  EXPECT_EQ(int64_vec[0], 10);
  EXPECT_EQ(int64_vec[1], 20);
  EXPECT_EQ(int64_vec[2], 30);
}

TEST(TypeConvertTest, ToInt32VecEmpty) {
  std::vector<int64_t> empty_vec;
  auto result = to_int32_vec(empty_vec);
  EXPECT_TRUE(result.empty());
}

TEST(TypeConvertTest, ToInt64VecEmpty) {
  std::vector<int32_t> empty_vec;
  auto result = to_int64_vec(empty_vec);
  EXPECT_TRUE(result.empty());
}

TEST(TypeConvertTest, SafeNarrowInt64ToInt32) {
  int64_t value = 42;
  int32_t result = safe_narrow<int32_t>(value);
  EXPECT_EQ(result, 42);
}

TEST(TypeConvertTest, SafeNarrowInt32ToInt16) {
  int32_t value = 1000;
  int16_t result = safe_narrow<int16_t>(value);
  EXPECT_EQ(result, 1000);
}

TEST(TypeConvertTest, ToInt32VecLargeValues) {
  std::vector<int64_t> int64_vec = {1000000, 2000000, 3000000};
  auto int32_vec = to_int32_vec(int64_vec);

  EXPECT_EQ(int32_vec.size(), 3);
  EXPECT_EQ(int32_vec[0], 1000000);
  EXPECT_EQ(int32_vec[1], 2000000);
  EXPECT_EQ(int32_vec[2], 3000000);
}

TEST(TypeConvertTest, ToInt64VecFromUint32) {
  std::vector<uint32_t> uint32_vec = {100, 200, 300};
  auto int64_vec = to_int64_vec(uint32_vec);

  EXPECT_EQ(int64_vec.size(), 3);
  EXPECT_EQ(int64_vec[0], 100);
  EXPECT_EQ(int64_vec[1], 200);
  EXPECT_EQ(int64_vec[2], 300);
}

} // namespace
} // namespace executorch::backends::aoti::slim
