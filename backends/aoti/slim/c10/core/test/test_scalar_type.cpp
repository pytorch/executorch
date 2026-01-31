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

// =============================================================================
// Test Data Structures for Parameterized Tests
// =============================================================================

struct ScalarTypeTestData {
  ScalarType dtype;
  int expected_enum_value;
  size_t expected_element_size;
  const char* expected_name;
  bool is_floating;
  bool is_integral;
  bool is_integral_with_bool;
  bool is_bool;
};

// All supported scalar types with their expected properties
const std::vector<ScalarTypeTestData> kAllScalarTypes = {
    // dtype, enum_value, element_size, name, is_float, is_int, is_int_w_bool,
    // is_bool
    {ScalarType::Char, 1, 1, "Char", false, true, true, false},
    {ScalarType::Short, 2, 2, "Short", false, true, true, false},
    {ScalarType::Int, 3, 4, "Int", false, true, true, false},
    {ScalarType::Long, 4, 8, "Long", false, true, true, false},
    {ScalarType::Float, 6, 4, "Float", true, false, false, false},
    {ScalarType::Bool, 11, 1, "Bool", false, false, true, true},
    {ScalarType::BFloat16, 15, 2, "BFloat16", true, false, false, false},
};

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

class ScalarTypeParamTest
    : public ::testing::TestWithParam<ScalarTypeTestData> {};

TEST_P(ScalarTypeParamTest, EnumValue) {
  const auto& data = GetParam();
  EXPECT_EQ(static_cast<int>(data.dtype), data.expected_enum_value)
      << "Failed for dtype: " << toString(data.dtype);
}

TEST_P(ScalarTypeParamTest, ElementSize) {
  const auto& data = GetParam();
  EXPECT_EQ(elementSize(data.dtype), data.expected_element_size)
      << "Failed for dtype: " << toString(data.dtype);
}

TEST_P(ScalarTypeParamTest, ToString) {
  const auto& data = GetParam();
  EXPECT_STREQ(toString(data.dtype), data.expected_name)
      << "Failed for dtype: " << toString(data.dtype);
}

TEST_P(ScalarTypeParamTest, IsFloatingType) {
  const auto& data = GetParam();
  EXPECT_EQ(isFloatingType(data.dtype), data.is_floating)
      << "Failed for dtype: " << toString(data.dtype);
}

TEST_P(ScalarTypeParamTest, IsIntegralTypeWithoutBool) {
  const auto& data = GetParam();
  EXPECT_EQ(isIntegralType(data.dtype, false), data.is_integral)
      << "Failed for dtype: " << toString(data.dtype);
}

TEST_P(ScalarTypeParamTest, IsIntegralTypeWithBool) {
  const auto& data = GetParam();
  EXPECT_EQ(isIntegralType(data.dtype, true), data.is_integral_with_bool)
      << "Failed for dtype: " << toString(data.dtype);
}

TEST_P(ScalarTypeParamTest, IsBoolType) {
  const auto& data = GetParam();
  EXPECT_EQ(isBoolType(data.dtype), data.is_bool)
      << "Failed for dtype: " << toString(data.dtype);
}

TEST_P(ScalarTypeParamTest, StreamOperator) {
  const auto& data = GetParam();
  std::ostringstream oss;
  oss << data.dtype;
  EXPECT_EQ(oss.str(), data.expected_name)
      << "Failed for dtype: " << toString(data.dtype);
}

INSTANTIATE_TEST_SUITE_P(
    AllTypes,
    ScalarTypeParamTest,
    ::testing::ValuesIn(kAllScalarTypes),
    [](const ::testing::TestParamInfo<ScalarTypeTestData>& info) {
      return std::string(info.param.expected_name);
    });

// =============================================================================
// Type Constant Tests
// =============================================================================

class ScalarTypeConstantsTest : public ::testing::Test {};

TEST_F(ScalarTypeConstantsTest, KCharConstant) {
  EXPECT_EQ(kChar, ScalarType::Char);
}

TEST_F(ScalarTypeConstantsTest, KShortConstant) {
  EXPECT_EQ(kShort, ScalarType::Short);
}

TEST_F(ScalarTypeConstantsTest, KIntConstant) {
  EXPECT_EQ(kInt, ScalarType::Int);
}

TEST_F(ScalarTypeConstantsTest, KLongConstant) {
  EXPECT_EQ(kLong, ScalarType::Long);
}

TEST_F(ScalarTypeConstantsTest, KFloatConstant) {
  EXPECT_EQ(kFloat, ScalarType::Float);
}

TEST_F(ScalarTypeConstantsTest, KBoolConstant) {
  EXPECT_EQ(kBool, ScalarType::Bool);
}

TEST_F(ScalarTypeConstantsTest, KBFloat16Constant) {
  EXPECT_EQ(kBFloat16, ScalarType::BFloat16);
}

// =============================================================================
// Edge Cases and Special Values
// =============================================================================

class ScalarTypeEdgeCasesTest : public ::testing::Test {};

TEST_F(ScalarTypeEdgeCasesTest, UndefinedToString) {
  EXPECT_STREQ(toString(ScalarType::Undefined), "Undefined");
}

TEST_F(ScalarTypeEdgeCasesTest, UndefinedIsNotFloating) {
  EXPECT_FALSE(isFloatingType(ScalarType::Undefined));
}

TEST_F(ScalarTypeEdgeCasesTest, UndefinedIsNotIntegral) {
  EXPECT_FALSE(isIntegralType(ScalarType::Undefined, false));
  EXPECT_FALSE(isIntegralType(ScalarType::Undefined, true));
}

TEST_F(ScalarTypeEdgeCasesTest, UndefinedIsNotBool) {
  EXPECT_FALSE(isBoolType(ScalarType::Undefined));
}

// =============================================================================
// Element Size Consistency Tests
// =============================================================================

class ElementSizeConsistencyTest : public ::testing::Test {};

TEST_F(ElementSizeConsistencyTest, CharMatchesSizeofInt8) {
  EXPECT_EQ(elementSize(ScalarType::Char), sizeof(int8_t));
}

TEST_F(ElementSizeConsistencyTest, ShortMatchesSizeofInt16) {
  EXPECT_EQ(elementSize(ScalarType::Short), sizeof(int16_t));
}

TEST_F(ElementSizeConsistencyTest, IntMatchesSizeofInt32) {
  EXPECT_EQ(elementSize(ScalarType::Int), sizeof(int32_t));
}

TEST_F(ElementSizeConsistencyTest, LongMatchesSizeofInt64) {
  EXPECT_EQ(elementSize(ScalarType::Long), sizeof(int64_t));
}

TEST_F(ElementSizeConsistencyTest, FloatMatchesSizeofFloat) {
  EXPECT_EQ(elementSize(ScalarType::Float), sizeof(float));
}

TEST_F(ElementSizeConsistencyTest, BoolMatchesSizeofBool) {
  EXPECT_EQ(elementSize(ScalarType::Bool), sizeof(bool));
}

TEST_F(ElementSizeConsistencyTest, BFloat16MatchesSizeofBFloat16) {
  EXPECT_EQ(elementSize(ScalarType::BFloat16), sizeof(BFloat16));
}
