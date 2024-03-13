/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpReluTest : public OperatorTest {
 protected:
  Tensor& op_relu_out(const Tensor& self, Tensor& out) {
    return torch::executor::aten::relu_outf(context_, self, out);
  }

  // Common testing for relu on two floating point Tensors.
  template <ScalarType DTYPE>
  void test_relu_execution_floats() {
    TensorFactory<DTYPE> tf;

    const std::vector<int32_t> sizes = {3, 2};

    Tensor in = tf.make(
        sizes, /*data=*/{-0.4775, 0.2948, -0.3984, 1.8690, -0.4048, 0.0});

    // Destination for the relu.
    Tensor out = tf.zeros(sizes);

    // Run relu.
    op_relu_out(in, out);

    // Check that it matches the expected output.
    EXPECT_TENSOR_EQ(
        out,
        tf.make(
            sizes,
            /*data=*/
            {0.0, 0.2948, 0.0, 1.8690, 0.0, 0.0}));
  }

  template <ScalarType DTYPE>
  void test_relu_execution_ints() {
    TensorFactory<DTYPE> tf;

    const std::vector<int32_t> sizes = {3, 2};

    Tensor in = tf.make(sizes, /*data=*/{-1, 2, 0, 3, 0, -5});

    // Destination for the relu.
    Tensor out = tf.zeros(sizes);

    // Run relu.
    op_relu_out(in, out);

    // Check that it matches the expected output.
    EXPECT_TENSOR_EQ(
        out,
        tf.make(
            sizes,
            /*data=*/
            {0, 2, 0, 3, 0, 0}));
  }
};

TEST_F(OpReluTest, FloatTensors) {
  test_relu_execution_floats<ScalarType::Float>();
}

TEST_F(OpReluTest, DoubleTensors) {
  test_relu_execution_floats<ScalarType::Double>();
}

TEST_F(OpReluTest, ByteTensors) {
  TensorFactory<ScalarType::Byte> tf;

  const std::vector<int32_t> sizes = {3, 2};

  Tensor in = tf.make(sizes, /*data=*/{1, 2, 0, 3, 0, 5});

  // Destination for the relu.
  Tensor out = tf.zeros(sizes);

  // Run relu.
  op_relu_out(in, out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_EQ(
      out,
      tf.make(
          sizes,
          /*data=*/
          {1, 2, 0, 3, 0, 5}));
}

TEST_F(OpReluTest, CharTensors) {
  test_relu_execution_ints<ScalarType::Char>();
}

TEST_F(OpReluTest, ShortTensors) {
  test_relu_execution_ints<ScalarType::Short>();
}

TEST_F(OpReluTest, IntTensors) {
  test_relu_execution_ints<ScalarType::Int>();
}

TEST_F(OpReluTest, LongTensors) {
  test_relu_execution_ints<ScalarType::Long>();
}

TEST_F(OpReluTest, InfAndNanPreserved) {
  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> sizes = {4, 2};

  Tensor in = tf.make(
      sizes,
      /*data=*/
      {-0.4775,
       0.2948,
       -0.3984,
       NAN,
       std::numeric_limits<float>::infinity(),
       -1 * std::numeric_limits<float>::infinity(),
       0.3,
       -0.4848});

  // Destination for the relu.
  Tensor out = tf.zeros(sizes);

  // Run full relu.
  op_relu_out(in, out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_EQ(
      out,
      tf.make(
          sizes,
          /*data=*/
          {0.0,
           0.2948,
           0.0,
           NAN,
           std::numeric_limits<float>::infinity(),
           0.0,
           0.3,
           0.0}));
}

TEST_F(OpReluTest, UnhandledDtypeDies) {
  // relu() doesn't handle Bool.
  TensorFactory<ScalarType::Bool> tf;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor a = tf.make(sizes, /*data=*/{false, true, false, true});

  // Destination for the relu.
  Tensor out = tf.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(context_, op_relu_out(a, out));
}

#if !defined(USE_ATEN_LIB)
TEST_F(OpReluTest, UpperBoundOutTensor) {
  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> sizes = {3, 2};

  Tensor in =
      tf.make(sizes, /*data=*/{-0.4775, 0.2948, -0.3984, 1.8690, -0.4048, 0.0});

  // Destination for the relu.
  Tensor out =
      tf.zeros({5, 7}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);

  // Run relu.
  op_relu_out(in, out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_EQ(
      out,
      tf.make(
          sizes,
          /*data=*/
          {0.0, 0.2948, 0.0, 1.8690, 0.0, 0.0}));
}
#endif

TEST_F(OpReluTest, SimpleGeneratedCase) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {10, 10},
      {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
  Tensor expected_result = tf.make(
      {10, 10},
      {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});

  Tensor out = tf.zeros({10, 10});
  Tensor ret = op_relu_out(x, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpReluTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.676039457321167,
       0.06196027994155884,
       0.36154472827911377,
       0.7953161001205444,
       0.7633233070373535,
       0.5809110999107361});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.676039457321167,
       0.06196027994155884,
       0.36154472827911377,
       0.7953161001205444,
       0.7633233070373535,
       0.5809110999107361});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_relu_out(x, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpReluTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.676039457321167,
       0.06196027994155884,
       0.36154472827911377,
       0.7953161001205444,
       0.7633233070373535,
       0.5809110999107361});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.676039457321167,
       0.06196027994155884,
       0.36154472827911377,
       0.7953161001205444,
       0.7633233070373535,
       0.5809110999107361});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_relu_out(x, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpReluTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Unbound dynamic shape not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.676039457321167,
       0.06196027994155884,
       0.36154472827911377,
       0.7953161001205444,
       0.7633233070373535,
       0.5809110999107361});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.676039457321167,
       0.06196027994155884,
       0.36154472827911377,
       0.7953161001205444,
       0.7633233070373535,
       0.5809110999107361});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = op_relu_out(x, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}
