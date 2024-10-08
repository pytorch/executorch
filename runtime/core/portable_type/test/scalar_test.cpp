/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/portable_type/scalar.h>
#include <executorch/test/utils/DeathTest.h>
#include <gtest/gtest.h>

using executorch::runtime::etensor::Scalar;

TEST(ScalarTest, ToScalarType) {
  Scalar s_d((double)3.141);
  EXPECT_EQ(s_d.to<double>(), 3.141);
  Scalar s_i((int64_t)3);
  EXPECT_EQ(s_i.to<int64_t>(), 3);
  Scalar s_b(true);
  EXPECT_EQ(s_b.to<bool>(), true);
}

TEST(ScalarTest, ToIntForFalseScalarPasses) {
  Scalar s_b(false);
  EXPECT_FALSE(s_b.isIntegral(/*includeBool=*/false));
  EXPECT_TRUE(s_b.isIntegral(/*includeBool=*/true));
  EXPECT_EQ(s_b.to<int64_t>(), 0);
}

TEST(ScalarTest, ToIntForTrueScalarPasses) {
  Scalar s_b(true);
  EXPECT_FALSE(s_b.isIntegral(/*includeBool=*/false));
  EXPECT_TRUE(s_b.isIntegral(/*includeBool=*/true));
  EXPECT_EQ(s_b.to<int64_t>(), 1);
}

TEST(ScalarTest, IntConstructor) {
  int int_val = 1;
  Scalar s_int(int_val);
  int32_t int32_val = 1;
  Scalar s_int32(int32_val);
  int64_t int64_val = 1;
  Scalar s_int64(int64_val);
  EXPECT_EQ(s_int.to<int64_t>(), s_int32.to<int64_t>());
  EXPECT_EQ(s_int32.to<int64_t>(), s_int64.to<int64_t>());
}
