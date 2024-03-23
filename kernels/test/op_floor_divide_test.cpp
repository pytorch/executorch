/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpFloorDivideTest : public OperatorTest {
 protected:
  Tensor&
  op_floor_divide_out(const Tensor& self, const Tensor& other, Tensor& out) {
    return torch::executor::aten::floor_divide_outf(context_, self, other, out);
  }

  template <ScalarType DTYPE>
  void test_integer_floor_divide() {
    TensorFactory<DTYPE> tf;

    const std::vector<int32_t> sizes = {3, 2};

    // Destination for the floor_divide.
    Tensor out = tf.zeros(sizes);

    // floor_divide two tensors.
    // Integer division of -8 / 6 return -1, but -8 // 6 is -2
    op_floor_divide_out(
        tf.make(sizes, /*data=*/{-8, 1, 2, 4, 8, 3}),
        tf.make(sizes, /*data=*/{6, 2, 2, 2, 2, -5}),
        out);

    // Check that it matches the expected output.
    EXPECT_TENSOR_EQ(out, tf.make(sizes, /*data=*/{-2, 0, 1, 2, 4, -1}));
  }

  template <ScalarType DTYPE>
  void test_floating_point_floor_divide() {
    TensorFactory<DTYPE> tf;

    const std::vector<int32_t> sizes = {3, 2};

    // Destination for the floor_divide.
    Tensor out = tf.zeros(sizes);

    // floor_divide two tensors.
    // std::floor(-0.5 / -0.1) == 5.0, but -0.5 // -0.1 yeilds 4.0
    op_floor_divide_out(
        tf.make(sizes, /*data=*/{-5.3, 1.1, 2.2, 4.4, 6.8, -0.5}),
        tf.make(sizes, /*data=*/{2.7, 2.0, 2.0, 2.0, 2.0, -0.1}),
        out);

    // Check that it matches the expected output.
    EXPECT_TENSOR_CLOSE(
        out, tf.make(sizes, /*data=*/{-2.0, 0.0, 1.0, 2.0, 3.0, 4.0}));
  }
};

// Common testing for floor-dividing two integer Tensors.
TEST_F(OpFloorDivideTest, ByteTensors) {
  TensorFactory<ScalarType::Byte> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the floor_divide.
  Tensor out = tf.zeros(sizes);

  // floor_divide two tensors.
  op_floor_divide_out(
      tf.make(sizes, /*data=*/{1, 2, 4, 8}),
      tf.make(sizes, /*data=*/{2, 2, 2, 2}),
      out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_EQ(out, tf.make(sizes, /*data=*/{0, 1, 2, 4}));
}

TEST_F(OpFloorDivideTest, CharTensors) {
  test_integer_floor_divide<ScalarType::Char>();
}

TEST_F(OpFloorDivideTest, ShortTensors) {
  test_integer_floor_divide<ScalarType::Short>();
}

TEST_F(OpFloorDivideTest, IntTensors) {
  test_integer_floor_divide<ScalarType::Int>();
}

TEST_F(OpFloorDivideTest, LongTensors) {
  test_integer_floor_divide<ScalarType::Long>();
}

// Common testing for floor-dividing two floating point Tensors.
TEST_F(OpFloorDivideTest, FloatTensors) {
  test_floating_point_floor_divide<ScalarType::Float>();
}

TEST_F(OpFloorDivideTest, DoubleTensors) {
  test_floating_point_floor_divide<ScalarType::Double>();
}

TEST_F(OpFloorDivideTest, UnhandledDtypeDies) {
  // floor_divide() doesn't handle Bool.
  TensorFactory<ScalarType::Bool> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Addends.
  Tensor a = tf.make(sizes, /*data=*/{false, true, false, true});
  Tensor b = tf.make(sizes, /*data=*/{true, true, true, true});

  // Destination for the foor_divide.
  Tensor out = tf.zeros(sizes);

  // Dividing the two boolean tensors should cause an assertion and kill the
  // test process.
  ET_EXPECT_KERNEL_FAILURE(context_, op_floor_divide_out(a, b, out));
}

// Mismatched shape tests.

TEST_F(OpFloorDivideTest, MismatchedInputShapesDies) {
  TensorFactory<ScalarType::Int> tf;

  // Addends with different shapes.
  Tensor a = tf.ones(/*sizes=*/{4});
  Tensor b = tf.ones(/*sizes=*/{2, 2});

  // Destination for the floor_divide; matches the shape of one of the inputs.
  Tensor out = tf.zeros(/*sizes=*/{4});

  // Adding the two mismatched tensors should cause an assertion and kill the
  // test process.
  ET_EXPECT_KERNEL_FAILURE(context_, op_floor_divide_out(a, b, out));
}

TEST_F(OpFloorDivideTest, MismatchedOutputShapesDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched output shape";
  }
  TensorFactory<ScalarType::Int> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Addends with the same shapes.
  Tensor a = tf.ones(sizes);
  Tensor b = tf.ones(sizes);

  // Destination with a different shape.
  Tensor out = tf.zeros(/*sizes=*/{4});

  // Adding the tensors into a mismatched output should cause an assertion and
  // kill the test process.
  ET_EXPECT_KERNEL_FAILURE(context_, op_floor_divide_out(a, b, out));
}

TEST_F(OpFloorDivideTest, BroadcastDimSizeIsOneAB) {
  GTEST_SKIP() << "Dynamic shape not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.6651028990745544,
       0.47241002321243286,
       0.15020078420639038,
       0.5280023813247681,
       0.9517974257469177,
       0.5294632911682129});
  Tensor y = tf.make({1, 2}, {0.522396445274353, 0.6753279566764832});
  Tensor expected_result = tf.make({3, 2}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_floor_divide_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpFloorDivideTest, BroadcastDimSizeMissingAB) {
  GTEST_SKIP() << "Dynamic shape not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.6651028990745544,
       0.47241002321243286,
       0.15020078420639038,
       0.5280023813247681,
       0.9517974257469177,
       0.5294632911682129});
  Tensor y = tf.make({2}, {0.522396445274353, 0.6753279566764832});
  Tensor expected_result = tf.make({3, 2}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_floor_divide_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpFloorDivideTest, BroadcastDimSizeIsOneBA) {
  GTEST_SKIP() << "Dynamic shape not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({1, 2}, {0.522396445274353, 0.6753279566764832});
  Tensor y = tf.make(
      {3, 2},
      {0.6651028990745544,
       0.47241002321243286,
       0.15020078420639038,
       0.5280023813247681,
       0.9517974257469177,
       0.5294632911682129});
  Tensor expected_result = tf.make({3, 2}, {0.0, 1.0, 3.0, 1.0, 0.0, 1.0});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_floor_divide_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpFloorDivideTest, BroadcastDimSizeMissingBA) {
  GTEST_SKIP() << "Dynamic shape not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({1, 2}, {0.522396445274353, 0.6753279566764832});
  Tensor y = tf.make(
      {3, 2},
      {0.6651028990745544,
       0.47241002321243286,
       0.15020078420639038,
       0.5280023813247681,
       0.9517974257469177,
       0.5294632911682129});
  Tensor expected_result = tf.make({3, 2}, {0.0, 1.0, 3.0, 1.0, 0.0, 1.0});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_floor_divide_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpFloorDivideTest, DynamicShapeUpperBoundSameAsExpected) {
  GTEST_SKIP() << "Dynamic shape not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.34620773792266846,
       0.7118645310401917,
       0.028005361557006836,
       0.8868894577026367,
       0.38272881507873535,
       0.19501900672912598});
  Tensor y = tf.make(
      {3, 2},
      {0.3282443881034851,
       0.7458182573318481,
       0.1568273901939392,
       0.6325231194496155,
       0.2777167558670044,
       0.09974533319473267});
  Tensor expected_result = tf.make({3, 2}, {1.0, 0.0, 0.0, 1.0, 1.0, 1.0});

  Tensor out =
      tf.zeros({3, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_floor_divide_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpFloorDivideTest, DynamicShapeUpperBoundLargerThanExpected) {
  GTEST_SKIP() << "Dynamic shape not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.34620773792266846,
       0.7118645310401917,
       0.028005361557006836,
       0.8868894577026367,
       0.38272881507873535,
       0.19501900672912598});
  Tensor y = tf.make(
      {3, 2},
      {0.3282443881034851,
       0.7458182573318481,
       0.1568273901939392,
       0.6325231194496155,
       0.2777167558670044,
       0.09974533319473267});
  Tensor expected_result = tf.make({3, 2}, {1.0, 0.0, 0.0, 1.0, 1.0, 1.0});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_floor_divide_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpFloorDivideTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.34620773792266846,
       0.7118645310401917,
       0.028005361557006836,
       0.8868894577026367,
       0.38272881507873535,
       0.19501900672912598});
  Tensor y = tf.make(
      {3, 2},
      {0.3282443881034851,
       0.7458182573318481,
       0.1568273901939392,
       0.6325231194496155,
       0.2777167558670044,
       0.09974533319473267});
  Tensor expected_result = tf.make({3, 2}, {1.0, 0.0, 0.0, 1.0, 1.0, 1.0});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = op_floor_divide_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}
