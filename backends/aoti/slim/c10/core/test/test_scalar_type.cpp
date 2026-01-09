/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>

#include <gtest/gtest.h>
#include <sstream>

using namespace executorch::backends::aoti::slim::c10;

class ScalarTypeTest : public ::testing::Test {};

TEST_F(ScalarTypeTest, FloatEnumValue) {
  // Verify Float has the correct enum value (6) to match PyTorch
  EXPECT_EQ(static_cast<int>(ScalarType::Float), 6);
}

TEST_F(ScalarTypeTest, KFloatConstant) {
  // Verify kFloat constant
  EXPECT_EQ(kFloat, ScalarType::Float);
}

TEST_F(ScalarTypeTest, ElementSizeFloat) {
  // Verify elementSize returns correct size for Float (4 bytes)
  EXPECT_EQ(elementSize(ScalarType::Float), sizeof(float));
  EXPECT_EQ(elementSize(ScalarType::Float), 4);
}

TEST_F(ScalarTypeTest, ToStringFloat) {
  // Verify toString returns correct string for Float
  EXPECT_STREQ(toString(ScalarType::Float), "Float");
}

TEST_F(ScalarTypeTest, ToStringUndefined) {
  // Verify toString returns correct string for Undefined
  EXPECT_STREQ(toString(ScalarType::Undefined), "Undefined");
}

TEST_F(ScalarTypeTest, IsFloatingType) {
  // Verify isFloatingType works correctly
  EXPECT_TRUE(isFloatingType(ScalarType::Float));
}

TEST_F(ScalarTypeTest, IsIntegralType) {
  // Verify isIntegralType works correctly
  // Currently no integral types are supported, so Float should return false
  EXPECT_FALSE(isIntegralType(ScalarType::Float, false));
  EXPECT_FALSE(isIntegralType(ScalarType::Float, true));
}

TEST_F(ScalarTypeTest, StreamOperator) {
  // Verify stream operator works
  std::ostringstream oss;
  oss << ScalarType::Float;
  EXPECT_EQ(oss.str(), "Float");
}
