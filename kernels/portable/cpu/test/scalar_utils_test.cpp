/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <gtest/gtest.h>

template <typename T1, typename T2>
struct promote_type_with_scalar_type_is_valid
    : std::integral_constant<
          bool,
          (std::is_same<T2, torch::executor::internal::B1>::value ||
           std::is_same<T2, torch::executor::internal::I8>::value ||
           std::is_same<T2, torch::executor::internal::F8>::value) &&
              !std::is_same<T1, exec_aten::BFloat16>::value &&
              !torch::executor::is_qint_type<T1>::value &&
              !torch::executor::is_bits_type<T1>::value> {};

template <typename T1, bool half_to_float>
struct CompileTimePromoteTypeWithScalarTypeTestCase {
  static void testAll() {
#define CALL_TEST_ONE(cpp_type, scalar_type) \
  testOne<                                   \
      cpp_type,                              \
      promote_type_with_scalar_type_is_valid<T1, cpp_type>::value>();
    ET_FORALL_SCALAR_TYPES(CALL_TEST_ONE)
#undef CALL_TEST_ONE
  }

  template <
      typename T2,
      bool valid,
      typename std::enable_if<valid, bool>::type = true>
  static void testOne() {
    auto actual = torch::executor::CppTypeToScalarType<
        typename torch::executor::native::utils::
            promote_type_with_scalar_type<T1, T2, half_to_float>::type>::value;
    const auto scalarType1 = torch::executor::CppTypeToScalarType<T1>::value;
    const auto scalarType2 = torch::executor::CppTypeToScalarType<T2>::value;
    T2 scalar_value = 0;
    auto expected = torch::executor::native::utils::promote_type_with_scalar(
        scalarType1, scalar_value, half_to_float);
    EXPECT_EQ(actual, expected)
        << "promoting " << (int)scalarType1 << " with " << (int)scalarType2
        << " given half_to_float = " << half_to_float << " expected "
        << (int)expected << " but got " << (int)actual;
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
#define INSTANTIATE_TYPE_TEST(cpp_type, scalar_type)                        \
  CompileTimePromoteTypeWithScalarTypeTestCase<cpp_type, false>::testAll(); \
  CompileTimePromoteTypeWithScalarTypeTestCase<cpp_type, true>::testAll();

  ET_FORALL_SCALAR_TYPES(INSTANTIATE_TYPE_TEST);
}
