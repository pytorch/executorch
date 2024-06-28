/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include <vector>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/test/utils/DeathTest.h>
#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::complex;
using exec_aten::ScalarType;

TEST(ScalarTypeUtilTest, ToString) {
  // Some known types.
  EXPECT_STREQ(torch::executor::toString(ScalarType::Int), "Int");
  EXPECT_STREQ(
      torch::executor::toString(ScalarType::ComplexHalf), "ComplexHalf");

  // Undefined, which is sort of a special case since it's not part of the
  // iteration macros but is still a part of the enum.
  EXPECT_STREQ(torch::executor::toString(ScalarType::Undefined), "Undefined");

  // Some out-of-range types, also demonstrating that NumOptions is not really a
  // scalar type.
  EXPECT_STREQ(
      torch::executor::toString(ScalarType::NumOptions), "UNKNOWN_SCALAR");
  EXPECT_STREQ(
      torch::executor::toString(static_cast<ScalarType>(127)),
      "UNKNOWN_SCALAR");
  EXPECT_STREQ(
      torch::executor::toString(static_cast<ScalarType>(-1)), "UNKNOWN_SCALAR");
}

TEST(ScalarTypeUtilTest, ElementSize) {
  struct TestCase {
    ScalarType type;
    size_t expected_size;
  };
  std::vector<TestCase> test_cases = {
      {ScalarType::Byte, sizeof(uint8_t)},
      {ScalarType::Char, sizeof(int8_t)},
      {ScalarType::Short, sizeof(int16_t)},
      {ScalarType::Int, sizeof(int32_t)},
      {ScalarType::Long, sizeof(int64_t)},
      {ScalarType::Half, sizeof(exec_aten::Half)},
      {ScalarType::Float, sizeof(float)},
      {ScalarType::Double, sizeof(double)},
      {ScalarType::ComplexHalf,
       sizeof(::exec_aten::complex<::exec_aten::Half>)},
      {ScalarType::ComplexFloat, sizeof(::exec_aten::complex<float>)},
      {ScalarType::ComplexDouble, sizeof(::exec_aten::complex<double>)},
      {ScalarType::Bool, sizeof(bool)},
      {ScalarType::QInt8, sizeof(::exec_aten::qint8)},
      {ScalarType::QUInt8, sizeof(::exec_aten::quint8)},
      {ScalarType::QInt32, sizeof(::exec_aten::qint32)},
      {ScalarType::BFloat16, sizeof(::exec_aten::BFloat16)},
      {ScalarType::QUInt4x2, sizeof(::exec_aten::quint4x2)},
      {ScalarType::QUInt2x4, sizeof(::exec_aten::quint2x4)},
  };
  for (const auto& test_case : test_cases) {
    EXPECT_EQ(
        torch::executor::elementSize(test_case.type), test_case.expected_size);
  }
}

TEST(ScalarTypeUtilTest, IsValidTrue) {
  // Some valid types.
  EXPECT_TRUE(torch::executor::isValid(ScalarType::Byte));
  EXPECT_TRUE(torch::executor::isValid(ScalarType::Float));
  EXPECT_TRUE(torch::executor::isValid(ScalarType::ComplexFloat));
  EXPECT_TRUE(torch::executor::isValid(ScalarType::Bits16));
}

TEST(ScalarTypeUtilTest, IsValidFalse) {
  // Undefined, which is sort of a special case since it's not part of the
  // iteration macros but is still a part of the enum.
  EXPECT_FALSE(torch::executor::isValid(ScalarType::Undefined));

  // Some out-of-range types, also demonstrating that NumOptions is not really a
  // scalar type.
  EXPECT_FALSE(torch::executor::isValid(ScalarType::NumOptions));
  EXPECT_FALSE(torch::executor::isValid(static_cast<ScalarType>(127)));
  EXPECT_FALSE(torch::executor::isValid(static_cast<ScalarType>(-1)));
}

TEST(ScalarTypeUtilTest, UnknownTypeElementSizeDies) {
  // Undefined, which is sort of a special case since it's not part of the
  // iteration macros but is still a part of the enum.
  ET_EXPECT_DEATH(torch::executor::elementSize(ScalarType::Undefined), "");

  // Some out-of-range types, also demonstrating that NumOptions is not really a
  // scalar type.
  ET_EXPECT_DEATH(torch::executor::elementSize(ScalarType::NumOptions), "");
  ET_EXPECT_DEATH(
      torch::executor::elementSize(static_cast<ScalarType>(127)), "");
  ET_EXPECT_DEATH(
      torch::executor::elementSize(static_cast<ScalarType>(-1)), "");
}

TEST(ScalarTypeUtilTest, canCastTest) {
  using exec_aten::ScalarType;
  using torch::executor::canCast;

  // Check some common cases

  // complex to non-complex fails
  ET_CHECK(!canCast(ScalarType::ComplexFloat, ScalarType::Float));
  ET_CHECK(!canCast(ScalarType::ComplexDouble, ScalarType::Double));

  // non-complex to complex is fine
  ET_CHECK(canCast(ScalarType::Float, ScalarType::ComplexFloat));
  ET_CHECK(canCast(ScalarType::Float, ScalarType::ComplexDouble));
  ET_CHECK(canCast(ScalarType::Int, ScalarType::ComplexDouble));

  // float to integral fails
  ET_CHECK(!canCast(ScalarType::Float, ScalarType::Int));
  ET_CHECK(!canCast(ScalarType::Double, ScalarType::Long));

  // integral to float in fine
  ET_CHECK(canCast(ScalarType::Int, ScalarType::Float));
  ET_CHECK(canCast(ScalarType::Long, ScalarType::Float));

  // non-bool to bool fails
  ET_CHECK(!canCast(ScalarType::Byte, ScalarType::Bool));
  ET_CHECK(!canCast(ScalarType::Int, ScalarType::Bool));

  // bool to non-bool is fine
  ET_CHECK(canCast(ScalarType::Bool, ScalarType::Byte));
  ET_CHECK(canCast(ScalarType::Bool, ScalarType::Int));
  ET_CHECK(canCast(ScalarType::Bool, ScalarType::Float));
}

TEST(ScalarTypeUtilTest, promoteTypesTest) {
  using exec_aten::ScalarType;
  using torch::executor::promoteTypes;

  // Check some common cases

  ET_CHECK(
      promoteTypes(ScalarType::Float, ScalarType::Double) ==
      ScalarType::Double);
  ET_CHECK(
      promoteTypes(ScalarType::Float, ScalarType::Short) == ScalarType::Float);

  ET_CHECK(
      promoteTypes(ScalarType::Float, ScalarType::Int) == ScalarType::Float);
  ET_CHECK(
      promoteTypes(ScalarType::Long, ScalarType::Float) == ScalarType::Float);

  ET_CHECK(
      promoteTypes(ScalarType::Bool, ScalarType::Bool) == ScalarType::Bool);

  ET_CHECK(promoteTypes(ScalarType::Byte, ScalarType::Int) == ScalarType::Int);
  ET_CHECK(
      promoteTypes(ScalarType::Char, ScalarType::Bool) == ScalarType::Char);
  ET_CHECK(promoteTypes(ScalarType::Bool, ScalarType::Int) == ScalarType::Int);
}

template <typename T1, typename T2>
struct promote_types_is_valid
    : std::integral_constant<
          bool,
          !std::is_same<T1, torch::executor::BFloat16>::value &&
              !std::is_same<T2, torch::executor::BFloat16>::value &&
              (std::is_same<T1, T2>::value ||
               (!torch::executor::is_qint_type<T1>::value &&
                !torch::executor::is_qint_type<T2>::value &&
                !torch::executor::is_bits_type<T1>::value &&
                !torch::executor::is_bits_type<T2>::value))> {};

template <typename T1, bool half_to_float>
struct CompileTimePromoteTypesTestCase {
  static void testAll() {
#define CALL_TEST_ONE(cpp_type, scalar_type) \
  testOne<cpp_type, promote_types_is_valid<T1, cpp_type>::value>();
    ET_FORALL_SCALAR_TYPES(CALL_TEST_ONE)
#undef CALL_TEST_ONE
  }

  template <
      typename T2,
      bool valid,
      typename std::enable_if<valid, bool>::type = true>
  static void testOne() {
    auto actual = torch::executor::CppTypeToScalarType<
        typename torch::executor::promote_types<T1, T2, half_to_float>::type>::
        value;
    const auto scalarType1 = torch::executor::CppTypeToScalarType<T1>::value;
    const auto scalarType2 = torch::executor::CppTypeToScalarType<T2>::value;
    auto expected = promoteTypes(scalarType1, scalarType2, half_to_float);
    EXPECT_EQ(actual, expected)
        << "promoting " << (int)scalarType1 << " to " << (int)scalarType2;
  }

  template <
      typename T2,
      bool valid,
      typename std::enable_if<!valid, bool>::type = true>
  static void testOne() {
    // Skip invalid case
  }
};

TEST(ScalarTypeUtilTest, compileTypePromoteTypesTest) {
#define INSTANTIATE_TYPE_TEST(cpp_type, scalar_type)           \
  CompileTimePromoteTypesTestCase<cpp_type, false>::testAll(); \
  CompileTimePromoteTypesTestCase<cpp_type, true>::testAll();

  ET_FORALL_SCALAR_TYPES(INSTANTIATE_TYPE_TEST);
}
