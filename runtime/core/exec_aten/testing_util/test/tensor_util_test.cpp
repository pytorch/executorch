/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <vector>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using exec_aten::TensorList;
using executorch::runtime::testing::IsCloseTo;
using executorch::runtime::testing::IsDataCloseTo;
using executorch::runtime::testing::IsDataEqualTo;
using executorch::runtime::testing::IsEqualTo;
using executorch::runtime::testing::IsListCloseTo;
using executorch::runtime::testing::IsListEqualTo;
using executorch::runtime::testing::tensor_data_is_close;
using executorch::runtime::testing::tensor_lists_are_close;
using executorch::runtime::testing::TensorFactory;
using executorch::runtime::testing::tensors_are_close;

// Exhaustively test all of our comparison functions every time. Also flip the
// params around to demonstrate that the underlying checks are commutative.

#define EXPECT_TENSORS_CLOSE_AND_EQUAL(t1__, t2__) \
  EXPECT_TRUE(tensors_are_close((t1__), (t2__)));  \
  EXPECT_TRUE(tensors_are_close((t2__), (t1__)));  \
  EXPECT_THAT((t1__), IsCloseTo((t2__)));          \
  EXPECT_THAT((t2__), IsCloseTo((t1__)));          \
  EXPECT_TENSOR_CLOSE((t1__), (t2__));             \
  EXPECT_TENSOR_CLOSE((t2__), (t1__));             \
  ASSERT_TENSOR_CLOSE((t1__), (t2__));             \
  ASSERT_TENSOR_CLOSE((t2__), (t1__));             \
  EXPECT_THAT((t1__), IsEqualTo(t2__));            \
  EXPECT_THAT((t2__), IsEqualTo(t1__));            \
  EXPECT_TENSOR_EQ((t1__), (t2__));                \
  EXPECT_TENSOR_EQ((t2__), (t1__));                \
  ASSERT_TENSOR_EQ((t1__), (t2__));                \
  ASSERT_TENSOR_EQ((t2__), (t1__))

#define EXPECT_TENSORS_CLOSE_BUT_NOT_EQUAL(t1__, t2__) \
  EXPECT_TRUE(tensors_are_close((t1__), (t2__)));      \
  EXPECT_TRUE(tensors_are_close((t2__), (t1__)));      \
  EXPECT_THAT((t1__), IsCloseTo((t2__)));              \
  EXPECT_THAT((t2__), IsCloseTo((t1__)));              \
  EXPECT_TENSOR_CLOSE((t1__), (t2__));                 \
  EXPECT_TENSOR_CLOSE((t2__), (t1__));                 \
  ASSERT_TENSOR_CLOSE((t1__), (t2__));                 \
  ASSERT_TENSOR_CLOSE((t2__), (t1__));                 \
  EXPECT_THAT((t1__), Not(IsEqualTo(t2__)));           \
  EXPECT_THAT((t2__), Not(IsEqualTo(t1__)));           \
  EXPECT_TENSOR_NE((t1__), (t2__));                    \
  EXPECT_TENSOR_NE((t2__), (t1__));                    \
  ASSERT_TENSOR_NE((t1__), (t2__));                    \
  ASSERT_TENSOR_NE((t2__), (t1__))

#define EXPECT_TENSORS_NOT_CLOSE_OR_EQUAL(t1__, t2__) \
  EXPECT_FALSE(tensors_are_close((t1__), (t2__)));    \
  EXPECT_FALSE(tensors_are_close((t2__), (t1__)));    \
  EXPECT_THAT((t1__), Not(IsCloseTo((t2__))));        \
  EXPECT_THAT((t2__), Not(IsCloseTo((t1__))));        \
  EXPECT_TENSOR_NOT_CLOSE((t1__), (t2__));            \
  EXPECT_TENSOR_NOT_CLOSE((t2__), (t1__));            \
  ASSERT_TENSOR_NOT_CLOSE((t1__), (t2__));            \
  ASSERT_TENSOR_NOT_CLOSE((t2__), (t1__));            \
  EXPECT_THAT((t1__), Not(IsEqualTo(t2__)));          \
  EXPECT_THAT((t2__), Not(IsEqualTo(t1__)));          \
  EXPECT_TENSOR_NE((t1__), (t2__));                   \
  EXPECT_TENSOR_NE((t2__), (t1__));                   \
  ASSERT_TENSOR_NE((t1__), (t2__));                   \
  ASSERT_TENSOR_NE((t2__), (t1__))

#define EXPECT_TENSORS_DATA_CLOSE_AND_EQUAL(t1__, t2__) \
  EXPECT_TRUE(tensor_data_is_close((t1__), (t2__)));    \
  EXPECT_TRUE(tensor_data_is_close((t2__), (t1__)));    \
  EXPECT_THAT((t1__), IsDataCloseTo((t2__)));           \
  EXPECT_THAT((t2__), IsDataCloseTo((t1__)));           \
  EXPECT_TENSOR_DATA_CLOSE((t1__), (t2__));             \
  EXPECT_TENSOR_DATA_CLOSE((t2__), (t1__));             \
  ASSERT_TENSOR_DATA_CLOSE((t1__), (t2__));             \
  ASSERT_TENSOR_DATA_CLOSE((t2__), (t1__));             \
  EXPECT_THAT((t1__), IsDataEqualTo(t2__));             \
  EXPECT_THAT((t2__), IsDataEqualTo(t1__));             \
  EXPECT_TENSOR_DATA_EQ((t1__), (t2__));                \
  EXPECT_TENSOR_DATA_EQ((t2__), (t1__));                \
  ASSERT_TENSOR_DATA_EQ((t1__), (t2__));                \
  ASSERT_TENSOR_DATA_EQ((t2__), (t1__))

#define EXPECT_TENSORS_DATA_CLOSE_BUT_NOT_EQUAL(t1__, t2__) \
  EXPECT_TRUE(tensor_data_is_close((t1__), (t2__)));        \
  EXPECT_TRUE(tensor_data_is_close((t2__), (t1__)));        \
  EXPECT_THAT((t1__), IsDataCloseTo((t2__)));               \
  EXPECT_THAT((t2__), IsDataCloseTo((t1__)));               \
  EXPECT_TENSOR_DATA_CLOSE((t1__), (t2__));                 \
  EXPECT_TENSOR_DATA_CLOSE((t2__), (t1__));                 \
  ASSERT_TENSOR_DATA_CLOSE((t1__), (t2__));                 \
  ASSERT_TENSOR_DATA_CLOSE((t2__), (t1__));                 \
  EXPECT_THAT((t1__), Not(IsDataEqualTo(t2__)));            \
  EXPECT_THAT((t2__), Not(IsDataEqualTo(t1__)));            \
  EXPECT_TENSOR_DATA_NE((t1__), (t2__));                    \
  EXPECT_TENSOR_DATA_NE((t2__), (t1__));                    \
  ASSERT_TENSOR_DATA_NE((t1__), (t2__));                    \
  ASSERT_TENSOR_DATA_NE((t2__), (t1__))

#define EXPECT_TENSORS_DATA_NOT_CLOSE_OR_EQUAL(t1__, t2__) \
  EXPECT_FALSE(tensor_data_is_close((t1__), (t2__)));      \
  EXPECT_FALSE(tensor_data_is_close((t2__), (t1__)));      \
  EXPECT_THAT((t1__), Not(IsDataCloseTo((t2__))));         \
  EXPECT_THAT((t2__), Not(IsDataCloseTo((t1__))));         \
  EXPECT_TENSOR_DATA_NOT_CLOSE((t1__), (t2__));            \
  EXPECT_TENSOR_DATA_NOT_CLOSE((t2__), (t1__));            \
  ASSERT_TENSOR_NOT_CLOSE((t1__), (t2__));                 \
  ASSERT_TENSOR_NOT_CLOSE((t2__), (t1__));                 \
  EXPECT_THAT((t1__), Not(IsDataEqualTo(t2__)));           \
  EXPECT_THAT((t2__), Not(IsDataEqualTo(t1__)));           \
  EXPECT_TENSOR_DATA_NE((t1__), (t2__));                   \
  EXPECT_TENSOR_DATA_NE((t2__), (t1__));                   \
  ASSERT_TENSOR_DATA_NE((t1__), (t2__));                   \
  ASSERT_TENSOR_DATA_NE((t2__), (t1__))

#define EXPECT_TENSOR_LISTS_CLOSE_AND_EQUAL(list1__, list2__) \
  EXPECT_TRUE(tensor_lists_are_close(                         \
      (list1__).data(),                                       \
      (list1__).size(),                                       \
      (list2__).data(),                                       \
      (list2__).size()));                                     \
  EXPECT_TRUE(tensor_lists_are_close(                         \
      (list2__).data(),                                       \
      (list2__).size(),                                       \
      (list1__).data(),                                       \
      (list1__).size()));                                     \
  EXPECT_THAT((list1__), IsListCloseTo((list2__)));           \
  EXPECT_THAT((list2__), IsListCloseTo((list1__)));           \
  EXPECT_TENSOR_LISTS_CLOSE((list1__), (list2__));            \
  EXPECT_TENSOR_LISTS_CLOSE((list2__), (list1__));            \
  ASSERT_TENSOR_LISTS_CLOSE((list1__), (list2__));            \
  ASSERT_TENSOR_LISTS_CLOSE((list2__), (list1__));            \
  EXPECT_THAT((list1__), IsListEqualTo(list2__));             \
  EXPECT_THAT((list2__), IsListEqualTo(list1__));             \
  EXPECT_TENSOR_LISTS_EQ((list1__), (list2__));               \
  EXPECT_TENSOR_LISTS_EQ((list2__), (list1__));               \
  ASSERT_TENSOR_LISTS_EQ((list1__), (list2__));               \
  ASSERT_TENSOR_LISTS_EQ((list2__), (list1__))

#define EXPECT_TENSOR_LISTS_CLOSE_BUT_NOT_EQUAL(list1__, list2__) \
  EXPECT_TRUE(tensor_lists_are_close(                             \
      (list1__).data(),                                           \
      (list1__).size(),                                           \
      (list2__).data(),                                           \
      (list2__).size()));                                         \
  EXPECT_TRUE(tensor_lists_are_close(                             \
      (list2__).data(),                                           \
      (list2__).size(),                                           \
      (list1__).data(),                                           \
      (list1__).size()));                                         \
  EXPECT_THAT((list1__), IsListCloseTo((list2__)));               \
  EXPECT_THAT((list2__), IsListCloseTo((list1__)));               \
  EXPECT_TENSOR_LISTS_CLOSE((list1__), (list2__));                \
  EXPECT_TENSOR_LISTS_CLOSE((list2__), (list1__));                \
  ASSERT_TENSOR_LISTS_CLOSE((list1__), (list2__));                \
  ASSERT_TENSOR_LISTS_CLOSE((list2__), (list1__));                \
  EXPECT_THAT((list1__), Not(IsListEqualTo(list2__)));            \
  EXPECT_THAT((list2__), Not(IsListEqualTo(list1__)));            \
  EXPECT_TENSOR_LISTS_NE((list1__), (list2__));                   \
  EXPECT_TENSOR_LISTS_NE((list2__), (list1__));                   \
  ASSERT_TENSOR_LISTS_NE((list1__), (list2__));                   \
  ASSERT_TENSOR_LISTS_NE((list2__), (list1__))

#define EXPECT_TENSOR_LISTS_NOT_CLOSE_OR_EQUAL(list1__, list2__) \
  EXPECT_FALSE(tensor_lists_are_close(                           \
      (list1__).data(),                                          \
      (list1__).size(),                                          \
      (list2__).data(),                                          \
      (list2__).size()));                                        \
  EXPECT_FALSE(tensor_lists_are_close(                           \
      (list2__).data(),                                          \
      (list2__).size(),                                          \
      (list1__).data(),                                          \
      (list1__).size()));                                        \
  EXPECT_THAT((list1__), Not(IsListCloseTo((list2__))));         \
  EXPECT_THAT((list2__), Not(IsListCloseTo((list1__))));         \
  EXPECT_TENSOR_LISTS_NOT_CLOSE((list1__), (list2__));           \
  EXPECT_TENSOR_LISTS_NOT_CLOSE((list2__), (list1__));           \
  ASSERT_TENSOR_LISTS_NOT_CLOSE((list1__), (list2__));           \
  ASSERT_TENSOR_LISTS_NOT_CLOSE((list2__), (list1__));           \
  EXPECT_THAT((list1__), Not(IsListEqualTo(list2__)));           \
  EXPECT_THAT((list2__), Not(IsListEqualTo(list1__)));           \
  EXPECT_TENSOR_LISTS_NE((list1__), (list2__));                  \
  EXPECT_TENSOR_LISTS_NE((list2__), (list1__));                  \
  ASSERT_TENSOR_LISTS_NE((list1__), (list2__));                  \
  ASSERT_TENSOR_LISTS_NE((list2__), (list1__))

namespace {
// calculate numel given size
int32_t size_to_numel(std::vector<int32_t> sizes) {
  int32_t numel = 1;
  for (auto size : sizes) {
    numel *= size;
  }
  return numel;
}

} // namespace

// Mismatched shapes/types/strides.

TEST(TensorUtilTest, DifferentDtypesAreNotCloseOrEqual) {
  // Create two tensors with identical shape and data, but different dtypes.
  TensorFactory<ScalarType::Int> tf_int;
  Tensor a = tf_int.make(/*sizes=*/{2, 2}, /*data=*/{1, 2, 4, 8});

  TensorFactory<ScalarType::Long> tf_long;
  Tensor b = tf_long.make(/*sizes=*/{2, 2}, /*data=*/{1, 2, 4, 8});

  EXPECT_TENSORS_NOT_CLOSE_OR_EQUAL(a, b);
}

TEST(TensorUtilTest, DifferentSizesAreNotCloseOrEqual) {
  TensorFactory<ScalarType::Int> tf;

  // Create two tensors with identical dtype and data, but different shapes.
  Tensor a = tf.make(/*sizes=*/{2, 2}, /*data=*/{1, 2, 4, 8});
  Tensor b = tf.make(/*sizes=*/{4}, /*data=*/{1, 2, 4, 8});

  EXPECT_TENSORS_NOT_CLOSE_OR_EQUAL(a, b);
}

TEST(TensorUtilTest, DifferentLayoutsDies) {
  TensorFactory<ScalarType::Int> tf;

  // Create two tensors with identical dtype, data and shapes, but different
  // strides.
  Tensor a = tf.make(
      /*sizes=*/{2, 2}, /*data=*/{1, 2, 4, 8}, /*strided=*/{1, 2});
  Tensor b = tf.make(
      /*sizes=*/{2, 2}, /*data=*/{1, 2, 4, 8}, /*strided=*/{2, 1});

  // Current `tensors_are_close` does not support comparing two tensors with
  // different stride.
  // TODO(T132992348): support comparison between tensors of different strides
  ET_EXPECT_DEATH(EXPECT_TENSORS_NOT_CLOSE_OR_EQUAL(a, b), "");
  ET_EXPECT_DEATH(EXPECT_TENSORS_CLOSE_AND_EQUAL(a, b), "");
}

// Int tensors, as a proxy for all non-floating-point types.

TEST(TensorUtilTest, IntTensorIsCloseAndEqualToItself) {
  TensorFactory<ScalarType::Int> tf;

  Tensor t = tf.make(/*sizes=*/{2, 2}, /*data=*/{1, 2, 4, 8});

  EXPECT_TENSORS_CLOSE_AND_EQUAL(t, t);
}

TEST(TensorUtilTest, IdenticalIntTensorsAreCloseAndEqual) {
  TensorFactory<ScalarType::Int> tf;

  // Create two tensors with identical shape, dtype, and data.
  Tensor a = tf.make(/*sizes=*/{2, 2}, /*data=*/{1, 2, 4, 8});
  Tensor b = tf.make(/*sizes=*/{2, 2}, /*data=*/{1, 2, 4, 8});

  EXPECT_TENSORS_CLOSE_AND_EQUAL(a, b);
}

TEST(TensorUtilTest, NonIdenticalIntTensorsAreNotCloseOrEqual) {
  TensorFactory<ScalarType::Int> tf;

  // Create two tensors with identical shape and dtype, but different data.
  Tensor a = tf.make(/*sizes=*/{2, 2}, /*data=*/{1, 2, 4, 8});
  Tensor b = tf.make(/*sizes=*/{2, 2}, /*data=*/{99, 2, 4, 8});

  EXPECT_TENSORS_NOT_CLOSE_OR_EQUAL(a, b);
}

TEST(TensorUtilTest, EmptyTensorsAreCloseAndEqual) {
  TensorFactory<ScalarType::Int> tf;

  // Create two tensors with identical shapes but no data.
  Tensor a = tf.make(/*sizes=*/{0, 2}, /*data=*/{});
  EXPECT_EQ(a.numel(), 0);
  EXPECT_EQ(a.nbytes(), 0);
  Tensor b = tf.make(/*sizes=*/{0, 2}, /*data=*/{});
  EXPECT_EQ(b.numel(), 0);
  EXPECT_EQ(b.nbytes(), 0);

  EXPECT_TENSORS_CLOSE_AND_EQUAL(a, b);
}

// Float tensors, as a proxy for all floating-point types.

TEST(TensorUtilTest, FloatTensorIsCloseAndEqualToItself) {
  TensorFactory<ScalarType::Float> tf;

  Tensor t = tf.make(/*sizes=*/{2, 2}, /*data=*/{1.1, 2.2, 4.4, 8.8});

  EXPECT_TENSORS_CLOSE_AND_EQUAL(t, t);
}

TEST(TensorUtilTest, IdenticalFloatTensorsAreCloseAndEqual) {
  TensorFactory<ScalarType::Float> tf;

  // Create two tensors with identical shape, dtype, and data.
  Tensor a = tf.make(/*sizes=*/{2, 2}, /*data=*/{1.1, 2.2, 4.4, 8.8});
  Tensor b = tf.make(/*sizes=*/{2, 2}, /*data=*/{1.1, 2.2, 4.4, 8.8});

  EXPECT_TENSORS_CLOSE_AND_EQUAL(a, b);
}

TEST(TensorUtilTest, NearlyIdenticalFloatTensorsAreCloseButNotEqual) {
  TensorFactory<ScalarType::Float> tf;

  // Create two tensors with identical shape and dtype, but slightly different
  // data.
  Tensor a = tf.make(/*sizes=*/{2, 2}, /*data=*/{1.1, 2.2, 4.4, 8.8});
  Tensor b = tf.make(
      /*sizes=*/{2, 2},
      {// First data element is slightly larger.
       std::nextafter(1.1f, 100.0f),
       // Remaining data elements are the same.
       2.2,
       4.4,
       8.8});

  EXPECT_TENSORS_CLOSE_BUT_NOT_EQUAL(a, b);
}

TEST(TensorUtilTest, NonIdenticalFloatTensorsAreNotCloseOrEqual) {
  TensorFactory<ScalarType::Float> tf;

  // Create two tensors with identical shape and dtype, but different data.
  Tensor a = tf.make(/*sizes=*/{2, 2}, /*data=*/{1.1, 2.2, 4.4, 8.8});
  Tensor b = tf.make(/*sizes=*/{2, 2}, /*data=*/{99.99, 2.2, 4.4, 8.8});

  EXPECT_TENSORS_NOT_CLOSE_OR_EQUAL(a, b);
}

TEST(TensorUtilTest, FloatNanElementsAreCloseAndEqual) {
  TensorFactory<ScalarType::Float> tf;

  // Two identical tensors with NaN elements.
  Tensor a = tf.make(/*sizes=*/{2, 2}, /*data=*/{1.1, NAN, 2.2, NAN});
  Tensor b = tf.make(/*sizes=*/{2, 2}, /*data=*/{1.1, NAN, 2.2, NAN});

  EXPECT_TENSORS_CLOSE_AND_EQUAL(a, b);
}

TEST(TensorUtilTest, FloatNanElementsAreNotEqualToNonNan) {
  TensorFactory<ScalarType::Float> tf;

  // Regression test ensuring that NaN elements are not compared equal to
  // non-NaN finite values.
  Tensor a = tf.make(/*sizes=*/{2, 2}, /*data=*/{1.1, NAN, 2.2, NAN});
  Tensor b = tf.make(/*sizes=*/{2, 2}, /*data=*/{1.1, 0.0, 2.2, 0.0});

  EXPECT_TENSORS_NOT_CLOSE_OR_EQUAL(a, b);
}

TEST(TensorUtilTest, FloatInfiniteElementsAreCloseAndEqual) {
  constexpr auto kInfinity = std::numeric_limits<float>::infinity();

  TensorFactory<ScalarType::Float> tf;

  // Two identical tensors with infinite elements.
  Tensor a =
      tf.make(/*sizes=*/{2, 2}, /*data=*/{-kInfinity, 1.1, 2.2, kInfinity});
  Tensor b =
      tf.make(/*sizes=*/{2, 2}, /*data=*/{-kInfinity, 1.1, 2.2, kInfinity});

  EXPECT_TENSORS_CLOSE_AND_EQUAL(a, b);
}

// Double: less test coverage since Float covers all of the branches, but
// demonstrate that it works.

TEST(TensorUtilTest, NearlyIdenticalDoubleTensorsAreCloseButNotEqual) {
  TensorFactory<ScalarType::Float> tf;

  // Create two tensors with identical shape and dtype, but slightly different
  // data.
  Tensor a = tf.make(/*sizes=*/{2, 2}, /*data=*/{1.1, 2.2, 4.4, 8.8});
  Tensor b = tf.make(
      /*sizes=*/{2, 2},
      {// First data element is slightly larger.
       std::nextafter(1.1f, 100.0f),
       // Remaining data elements are the same.
       2.2,
       4.4,
       8.8});

  EXPECT_TENSORS_CLOSE_BUT_NOT_EQUAL(a, b);
}

TEST(TensorUtilTest, DoubleAndInfinitNanElementsAreCloseAndEqual) {
  constexpr auto kInfinity = std::numeric_limits<double>::infinity();

  TensorFactory<ScalarType::Double> tf;

  // Two identical tensors with NaN and infinite elements.
  Tensor a =
      tf.make(/*sizes=*/{2, 2}, /*data=*/{-kInfinity, NAN, 1.1, kInfinity});
  Tensor b =
      tf.make(/*sizes=*/{2, 2}, /*data=*/{-kInfinity, NAN, 1.1, kInfinity});

  EXPECT_TENSORS_CLOSE_AND_EQUAL(a, b);
}

// Testing closeness with tolerances

TEST(TensorUtilTest, TensorsAreCloseWithTol) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Double> td;

  // Create two tensors with identical shape and dtype, but different data.
  Tensor af = tf.make(/*sizes=*/{2, 2}, /*data=*/{1.0, 2.099999, 0.0, -0.05});
  Tensor bf = tf.make(/*sizes=*/{2, 2}, /*data=*/{1.099999, 2.0, 0.05, 0.0});

  EXPECT_TENSOR_CLOSE_WITH_TOL(af, bf, 0.0, 0.1);

  // Create two tensors with identical shape and dtype, but different data.
  Tensor ad = td.make(/*sizes=*/{2, 2}, /*data=*/{1.099, 2.199, NAN, -9.0});
  Tensor bd = td.make(/*sizes=*/{2, 2}, /*data=*/{1.0, 2.0, NAN, -10.0});

  EXPECT_TENSOR_CLOSE_WITH_TOL(ad, bd, 0.1, 0.0);
}

TEST(TensorUtilTest, TensorsAreNotCloseWithTol) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Double> td;

  // Create two tensors with identical shape and dtype, but different data.
  Tensor af = tf.make(/*sizes=*/{3}, /*data=*/{1.00, NAN, -10.0});
  Tensor bf = tf.make(/*sizes=*/{3}, /*data=*/{1.11, NAN, -10.0});

  EXPECT_TENSOR_NOT_CLOSE_WITH_TOL(af, bf, 0.0, 0.1);

  // Create two tensors with identical shape and dtype, but different data.
  Tensor ad = td.make(/*sizes=*/{3}, /*data=*/{1.0, 0.0, -10.0});
  Tensor bd = td.make(/*sizes=*/{3}, /*data=*/{1.0, 0.0, -9.0});

  EXPECT_TENSOR_NOT_CLOSE_WITH_TOL(ad, bd, 0.1, 0.0);

  // Create two tensors with identical shape and dtype, but different data.
  ad = tf.make(/*sizes=*/{3}, /*data=*/{1.0, 2.0, 0.00001});
  bd = tf.make(/*sizes=*/{3}, /*data=*/{1.0, 2.0, 0.0});

  EXPECT_TENSOR_NOT_CLOSE_WITH_TOL(ad, bd, 0.1, 0.0);
}

//
// Tests for shape-agnostic data equality.
//

// Common testing for EXPECT_TENSOR_DATA_EQ in different input sizes and
// dtypes.
template <ScalarType DTYPE>
void test_data_equal(
    std::vector<int32_t> t1_sizes,
    std::vector<int32_t> t2_sizes) {
  TensorFactory<DTYPE> tf;

  // get corresponding ctype for input DTYPEs
  using ctype = typename TensorFactory<DTYPE>::ctype;

  // Get the size of data of t1 and t2.
  // Make sure the two sizes are equal.
  auto numel = size_to_numel(t1_sizes);
  ASSERT_EQ(numel, size_to_numel(t2_sizes));

  // Set up data vector for t1 and t2.
  // Set them as a same random vector to test them generally.
  std::vector<ctype> t1_data(numel);
  std::generate(t1_data.begin(), t1_data.end(), std::rand);
  std::vector<ctype> t2_data(numel);
  t2_data = t1_data;

  Tensor t1 = tf.make(t1_sizes, t1_data);
  Tensor t2 = tf.make(t2_sizes, t2_data);

  EXPECT_TENSORS_DATA_CLOSE_AND_EQUAL(t1, t2);
}

TEST(TensorUtilTest, TensorDataEqualSizeEqual) {
#define TEST_ENTRY(ctype, dtype) \
  test_data_equal<ScalarType::dtype>({3, 4, 5}, {3, 4, 5});
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(TensorUtilTest, TensorDataEqualSizeUnequal) {
#define TEST_ENTRY(ctype, dtype) \
  test_data_equal<ScalarType::dtype>({3, 4, 5}, {3, 5, 4});
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(TensorUtilTest, EmptyTensorsSupported) {
#define TEST_ENTRY(ctype, dtype) \
  test_data_equal<ScalarType::dtype>({3, 4, 0, 5}, {3, 4, 0, 5});
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(TensorUtilTest, ZeroDimTensorsSupported) {
#define TEST_ENTRY(ctype, dtype) test_data_equal<ScalarType::dtype>({}, {});
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

// Common testing for EXPECT_TENSOR_DATA_CLOSE in different input sizes and
// dtypes.
template <
    ScalarType DTYPE,
    std::enable_if_t<
        DTYPE == ScalarType::Float || DTYPE == ScalarType::Double,
        bool> = true>
void test_data_close_but_not_equal(
    std::vector<int32_t> t1_sizes,
    std::vector<int32_t> t2_sizes) {
  TensorFactory<DTYPE> tf;

  // get corresponding ctype for input DTYPEs
  using ctype = typename TensorFactory<DTYPE>::ctype;

  // get the size of data of t1 and t2
  // make sure the two sizes are equal
  auto numel = size_to_numel(t1_sizes);
  ASSERT_EQ(numel, size_to_numel(t2_sizes));

  // set up data vector for t1 and t2
  // set them as a almost same random vector (only the first element are
  // different if the first element exists) to test them generally
  std::vector<ctype> t1_data(numel);
  std::generate(t1_data.begin(), t1_data.end(), std::rand);
  std::vector<ctype> t2_data(numel);
  t2_data = t1_data;

  // Set the first element of t2 slightly larger than 0
  // the "first element" only work if t2_numel > 0
  // The checking with ctype.max() is to prevent overflow
  if (numel > 0) {
    if (t2_data[0] < std::numeric_limits<ctype>::max() - 100) {
      t2_data[0] = std::nextafter(t2_data[0], t2_data[0] + 100.0f);
    } else {
      t2_data[0] = std::nextafter(t2_data[0], t2_data[0] - 100.f);
    }
  }

  Tensor t1 = tf.make(t1_sizes, t1_data);
  Tensor t2 = tf.make(t2_sizes, t2_data);

  EXPECT_TENSORS_DATA_CLOSE_AND_EQUAL(t1, t2);
}

TEST(TensorUtilTest, TensorDataCloseNotEqualSizeEqual) {
#define TEST_ENTRY(ctype, dtype) \
  test_data_equal<ScalarType::dtype>({3, 4, 5}, {3, 4, 5});
  ET_FORALL_FLOAT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(TensorUtilTest, TensorDataCloseNotEqualSizeUnequal) {
#define TEST_ENTRY(ctype, dtype) \
  test_data_equal<ScalarType::dtype>({3, 4, 5}, {3, 5, 4});
  ET_FORALL_FLOAT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

// Common testing for EXPECT_TENSOR_DATA_EQ in different input sizes and
// dtypes.
template <ScalarType DTYPE_T1, ScalarType DTYPE_T2>
void test_data_equal_but_size_or_dtype_mismatch(
    std::vector<int32_t> t1_sizes,
    std::vector<int32_t> t2_sizes) {
  TensorFactory<DTYPE_T1> tf_t1;
  TensorFactory<DTYPE_T2> tf_t2;

  Tensor t1 = tf_t1.zeros(t1_sizes);
  Tensor t2 = tf_t2.zeros(t2_sizes);

  EXPECT_TENSORS_DATA_NOT_CLOSE_OR_EQUAL(t1, t2);
}

TEST(TensorUtilTest, TensorDataTypeMismatched) {
  std::vector<int32_t> sizes = {3, 4, 5, 6};
  test_data_equal_but_size_or_dtype_mismatch<
      ScalarType::Float,
      ScalarType::Double>(sizes, sizes);
  test_data_equal_but_size_or_dtype_mismatch<
      ScalarType::Int,
      ScalarType::Double>(sizes, sizes);
}

TEST(TensorUtilTest, TensorSizeMismatched) {
  std::vector<int32_t> sizes_t1 = {3, 4, 5, 6};
  std::vector<int32_t> sizes_t2 = {3, 4, 5, 7};
  test_data_equal_but_size_or_dtype_mismatch<
      ScalarType::Float,
      ScalarType::Float>(sizes_t1, sizes_t2);
}

TEST(TensorUtilTest, TensorDataMismatched) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t1 = tf.make(/*size=*/{3, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  Tensor t2 = tf.make(/*size=*/{3, 2}, /*data=*/{1, 2, 3, 1, 5, 6});
  Tensor t3 = tf.make(/*size=*/{2, 3}, /*data=*/{1, 2, 3, 1, 5, 6});
  EXPECT_TENSORS_DATA_NOT_CLOSE_OR_EQUAL(t1, t2);
  EXPECT_TENSORS_DATA_NOT_CLOSE_OR_EQUAL(t1, t3);

  Tensor t_zero_dim = tf.make(/*size=*/{}, /*data=*/{0});
  Tensor t_empty = tf.make(/*size=*/{0}, /*data=*/{});
  EXPECT_TENSORS_DATA_NOT_CLOSE_OR_EQUAL(t_zero_dim, t_empty);
}

// Testing data closeness with tolerances

TEST(TensorUtilTest, TensorDataCloseWithTol) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Double> td;

  // Create two tensors with identical shape and dtype, but different data.
  Tensor af = tf.make(/*sizes=*/{4, 1}, /*data=*/{1.0, 2.099, 0.0, -0.05});
  Tensor bf = tf.make(/*sizes=*/{2, 2}, /*data=*/{1.099, 2.0, 0.05, 0.0});

  EXPECT_TENSOR_DATA_CLOSE_WITH_TOL(af, bf, 0.0, 0.1);

  // Create two tensors with identical shape and dtype, but different data.
  Tensor ad = td.make(/*sizes=*/{2, 2}, /*data=*/{1.099, 2.199, NAN, -9.0});
  Tensor bd = td.make(/*sizes=*/{4}, /*data=*/{1.0, 2.0, NAN, -10.0});

  EXPECT_TENSOR_DATA_CLOSE_WITH_TOL(ad, bd, 0.1, 0.0);
}

TEST(TensorUtilTest, TensorDataNotCloseWithTol) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Double> td;

  // Create two tensors with identical shape and dtype, but different data.
  Tensor af = tf.make(/*sizes=*/{3}, /*data=*/{1.00, 0.0, -10.0});
  Tensor bf = tf.make(/*sizes=*/{3, 1}, /*data=*/{1.11, 0.0, -10.0});

  EXPECT_TENSOR_DATA_NOT_CLOSE_WITH_TOL(af, bf, 0.0, 0.1);

  // Create two tensors with identical shape and dtype, but different data.
  Tensor ad = td.make(/*sizes=*/{2, 2}, /*data=*/{1.0, 0.0, -10.0, 0.0});
  Tensor bd = td.make(/*sizes=*/{4}, /*data=*/{1.0, 0.0, -9.0, 0.0});

  EXPECT_TENSOR_DATA_NOT_CLOSE_WITH_TOL(ad, bd, 0.1, 0.0);

  // Create two tensors with identical shape and dtype, but different data.
  ad = tf.make(/*sizes=*/{1, 4}, /*data=*/{1.0, 2.0, NAN, 0.00001});
  bd = tf.make(/*sizes=*/{2, 2}, /*data=*/{1.0, 2.0, NAN, 0.0});

  EXPECT_TENSOR_DATA_NOT_CLOSE_WITH_TOL(ad, bd, 0.1, 0.0);
}

//
// Tests for TensorList helpers.
//

TEST(TensorUtilTest, TensorListsCloseAndEqual) {
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Float> tf_float;

  // Two lists of tensors that should be close and equal. Elements have
  // different shapes and dtypes.
  std::vector<Tensor> vec1 = {
      tf_int.zeros(/*sizes=*/{1, 2}),
      tf_float.ones(/*sizes=*/{2, 1}),
  };
  TensorList list1(vec1.data(), vec1.size());
  std::vector<Tensor> vec2 = {
      tf_int.zeros(/*sizes=*/{1, 2}),
      tf_float.ones(/*sizes=*/{2, 1}),
  };
  TensorList list2(vec2.data(), vec2.size());

  // Show that we can compare a mix of vectors and TensorLists.
  EXPECT_TENSOR_LISTS_CLOSE_AND_EQUAL(list1, list2);
  EXPECT_TENSOR_LISTS_CLOSE_AND_EQUAL(vec1, list2);
  EXPECT_TENSOR_LISTS_CLOSE_AND_EQUAL(list1, vec2);
  EXPECT_TENSOR_LISTS_CLOSE_AND_EQUAL(vec1, vec2);
}

TEST(TensorUtilTest, EmptyTensorListsAreCloseAndEqual) {
  // Two empty lists.
  TensorList list1;
  EXPECT_EQ(list1.size(), 0);
  TensorList list2;
  EXPECT_EQ(list2.size(), 0);

  EXPECT_TENSOR_LISTS_CLOSE_AND_EQUAL(list1, list2);
}

TEST(TensorUtilTest, TensorListsCloseButNotEqual) {
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Float> tf_float;

  // Two lists of tensors that should be close and equal. Elements have
  // different shapes and dtypes.
  std::vector<Tensor> vec1 = {
      tf_int.zeros(/*sizes=*/{1, 2}),
      tf_float.ones(/*sizes=*/{2, 1}),
  };
  TensorList list1(vec1.data(), vec1.size());
  std::vector<Tensor> vec2 = {
      tf_int.zeros(/*sizes=*/{1, 2}),
      tf_float.ones(/*sizes=*/{2, 1}),
  };
  TensorList list2(vec2.data(), vec2.size());

  // Tweak a float value slightly.
  vec1[1].mutable_data_ptr<float>()[0] = std::nextafter(1.0f, 100.0f);

  // Show that we can compare a mix of vectors and TensorLists.
  EXPECT_TENSOR_LISTS_CLOSE_BUT_NOT_EQUAL(list1, list2);
}

TEST(TensorUtilTest, TensorListsWithDifferentDataAreNotCloseOrEqual) {
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Float> tf_float;

  std::vector<Tensor> vec1 = {
      tf_int.zeros(/*sizes=*/{1, 2}),
      tf_float.ones(/*sizes=*/{2, 1}),
  };
  TensorList list1(vec1.data(), vec1.size());

  std::vector<Tensor> vec2 = {
      tf_int.zeros(/*sizes=*/{1, 2}),
      tf_float.zeros(/*sizes=*/{2, 1}), // vs. ones() in the first list.
  };
  TensorList list2(vec2.data(), vec2.size());

  EXPECT_TENSOR_LISTS_NOT_CLOSE_OR_EQUAL(list1, list2);
}

TEST(TensorUtilTest, TensorListsWithDifferentLengthsAreNotCloseOrEqual) {
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Float> tf_float;

  std::vector<Tensor> vec1 = {
      tf_int.zeros(/*sizes=*/{1, 2}),
      tf_float.ones(/*sizes=*/{2, 1}),
  };
  TensorList list1(vec1.data(), vec1.size());

  std::vector<Tensor> vec2 = {
      tf_int.zeros(/*sizes=*/{1, 2}),
      // Missing second element.
  };
  TensorList list2(vec2.data(), vec2.size());

  EXPECT_TENSOR_LISTS_NOT_CLOSE_OR_EQUAL(list1, list2);
}

// We don't need to test the ATen operator<<() implementations since they're
// tested elsewhere.
#ifndef USE_ATEN_LIB

// Printing/formatting helpers.

TEST(TensorUtilTest, ScalarTypeStreamSmokeTest) {
  // Don't test everything, since operator<<(ScalarType) is just a wrapper
  // around a separately-tested function. Just demonstrate that the stream
  // wrapper works, and gives us a little more information for unknown types.
  {
    std::stringstream out;
    out << ScalarType::Byte;
    EXPECT_STREQ(out.str().c_str(), "Byte");
  }
  {
    std::stringstream out;
    out << static_cast<ScalarType>(127);
    EXPECT_STREQ(out.str().c_str(), "Unknown(127)");
  }
}

TEST(TensorUtilTest, TensorStreamInt) {
  TensorFactory<ScalarType::Int> tf;

  Tensor t = tf.make(/*sizes=*/{2, 2}, /*data=*/{1, 2, 4, 8});

  std::stringstream out;
  out << t;
  EXPECT_STREQ(
      out.str().c_str(), "ETensor(sizes={2, 2}, dtype=Int, data={1, 2, 4, 8})");
}

TEST(TensorUtilTest, TensorStreamDouble) {
  TensorFactory<ScalarType::Double> tf;

  Tensor t = tf.make(/*sizes=*/{2, 2}, /*data=*/{1.1, 2.2, 4.4, 8.8});

  std::stringstream out;
  out << t;
  EXPECT_STREQ(
      out.str().c_str(),
      "ETensor(sizes={2, 2}, dtype=Double, data={1.1, 2.2, 4.4, 8.8})");
}

TEST(TensorUtilTest, TensorStreamBool) {
  TensorFactory<ScalarType::Bool> tf;

  Tensor t = tf.make(/*sizes=*/{2, 2}, /*data=*/{true, false, true, false});

  std::stringstream out;
  out << t;
  EXPECT_STREQ(
      out.str().c_str(),
      "ETensor(sizes={2, 2}, dtype=Bool, data={1, 0, 1, 0})");
}

#endif // !USE_ATEN_LIB
