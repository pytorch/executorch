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

class OpMinimumOutTest : public OperatorTest {
 protected:
  Tensor& op_minimum_out(const Tensor& self, const Tensor& other, Tensor& out) {
    return torch::executor::aten::minimum_outf(context_, self, other, out);
  }

  // Common testing for minimum operator
  template <ScalarType DTYPE>
  void test_minimum_out_same_size() {
    TensorFactory<DTYPE> tf;
    const std::vector<int32_t> sizes = {2, 2};

    // Destination for the minimum operator.
    Tensor out = tf.zeros(sizes);

    op_minimum_out(
        tf.make(sizes, /*data=*/{1, 2, 4, 8}),
        tf.make(sizes, /*data=*/{3, 0, 4, 9}),
        out);

    // Check that it matches to the expected output.
    EXPECT_TENSOR_EQ(out, tf.make(sizes, /*data=*/{1, 0, 4, 8}));
  }
};

TEST_F(OpMinimumOutTest, ByteTensors) {
  test_minimum_out_same_size<ScalarType::Byte>();
}

TEST_F(OpMinimumOutTest, CharTensors) {
  test_minimum_out_same_size<ScalarType::Char>();
}

TEST_F(OpMinimumOutTest, ShortTensors) {
  test_minimum_out_same_size<ScalarType::Short>();
}

TEST_F(OpMinimumOutTest, IntTensors) {
  test_minimum_out_same_size<ScalarType::Int>();
}

TEST_F(OpMinimumOutTest, LongTensors) {
  test_minimum_out_same_size<ScalarType::Long>();
}

TEST_F(OpMinimumOutTest, HalfTensors) {
  test_minimum_out_same_size<ScalarType::Half>();
}

TEST_F(OpMinimumOutTest, FloatTensors) {
  test_minimum_out_same_size<ScalarType::Float>();
}

TEST_F(OpMinimumOutTest, DoubleTensors) {
  test_minimum_out_same_size<ScalarType::Double>();
}

TEST_F(OpMinimumOutTest, BothScalarTensors) {
  // Checks the case when both cases are scalar.
  TensorFactory<ScalarType::Float> tf;
  const std::vector<int32_t> sizes = {1, 1};
  Tensor out = tf.zeros(sizes);
  op_minimum_out(tf.make(sizes, {1.2}), tf.make(sizes, {3.5}), out);
  EXPECT_TENSOR_EQ(out, tf.make(sizes, {1.2}));
}

TEST_F(OpMinimumOutTest, LeftScalarTensor) {
  // Checks the case where one of the tensor is a singleton tensor.

  TensorFactory<ScalarType::Float> tf;
  const std::vector<int32_t> sizes_1 = {1, 1};
  const std::vector<int32_t> sizes_2 = {2, 2};
  Tensor out1 = tf.zeros(sizes_2);
  Tensor out2 = tf.zeros(sizes_2);

  auto a = tf.make(sizes_1, /*data=*/{1.0});
  auto b = tf.make(sizes_2, /*data=*/{3.5, -1.0, 0.0, 5.5});

  // Case 1 : First argument is singleton.
  op_minimum_out(a, b, out1);
  EXPECT_TENSOR_EQ(out1, tf.make(sizes_2, {1.0, -1.0, 0.0, 1.0}));

  // Case 2: Second argument is singleton
  op_minimum_out(b, a, out2);
  EXPECT_TENSOR_EQ(out2, tf.make(sizes_2, {1.0, -1.0, 0.0, 1.0}));
}

TEST_F(OpMinimumOutTest, MismatchedInputShapesDies) {
  // First and second argument have different shape
  TensorFactory<ScalarType::Float> tf;
  Tensor out = tf.zeros({2, 2});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_minimum_out(tf.ones({2, 2}), tf.ones({3, 3}), out));
}

TEST_F(OpMinimumOutTest, MismatchedOutputShapesDies) {
  // First and second argument have same shape, but output has different shape.
  TensorFactory<ScalarType::Float> tf;
  Tensor out = tf.zeros({3, 3});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_minimum_out(tf.ones({2, 2}), tf.ones({3, 3}), out));
}

TEST_F(OpMinimumOutTest, MismatchedOutputShapeWithSingletonDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched output shape";
  }
  // First argument is singleton but second and output has different shape.
  TensorFactory<ScalarType::Float> tf;
  Tensor out = tf.zeros({4, 4});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_minimum_out(tf.ones({1, 1}), tf.ones({3, 3}), out));
}

/* %python
import torch
torch.manual_seed(0)
x = torch.rand(3, 2)
y = torch.rand(3, 2)
res = torch.minimum(x, y)
op = "op_minimum_out"
dtype = "ScalarType::Float"
check = "EXPECT_TENSOR_EQ" */

TEST_F(OpMinimumOutTest, DynamicShapeUpperBoundSameAsExpected) {
  /* %python
  out_args = "{3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(binary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636});
  Tensor y = tf.make(
      {3, 2},
      {0.4900934100151062,
       0.8964447379112244,
       0.455627977848053,
       0.6323062777519226,
       0.3488934636116028,
       0.40171730518341064});
  Tensor expected = tf.make(
      {3, 2},
      {0.4900934100151062,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.40171730518341064});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_minimum_out(x, y, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpMinimumOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(binary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636});
  Tensor y = tf.make(
      {3, 2},
      {0.4900934100151062,
       0.8964447379112244,
       0.455627977848053,
       0.6323062777519226,
       0.3488934636116028,
       0.40171730518341064});
  Tensor expected = tf.make(
      {3, 2},
      {0.4900934100151062,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.40171730518341064});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_minimum_out(x, y, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpMinimumOutTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND"
  %rewrite(binary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636});
  Tensor y = tf.make(
      {3, 2},
      {0.4900934100151062,
       0.8964447379112244,
       0.455627977848053,
       0.6323062777519226,
       0.3488934636116028,
       0.40171730518341064});
  Tensor expected = tf.make(
      {3, 2},
      {0.4900934100151062,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.40171730518341064});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  op_minimum_out(x, y, out);
  EXPECT_TENSOR_EQ(out, expected);
}
