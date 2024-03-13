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

class OpRoundTest : public OperatorTest {
 protected:
  Tensor& op_round_out(const Tensor& self, Tensor& out) {
    return torch::executor::aten::round_outf(context_, self, out);
  }

  // Common testing for round on two floating point Tensors.
  template <ScalarType DTYPE>
  void test_round_execution_floats() {
    TensorFactory<DTYPE> tf;

    const std::vector<int32_t> sizes = {11};

    Tensor in = tf.make(
        sizes,
        /*data=*/{1.5, -1.5, 0, 1.5, 2.5, 3.5, 4.5, 1.4, -1.4, 1.7, -1.7});

    // Destination for the round.
    Tensor out = tf.zeros(sizes);

    // Run round.
    op_round_out(in, out);

    // Check that it matches the expected output.
    EXPECT_TENSOR_EQ(
        out,
        tf.make(
            sizes,
            /*data=*/
            {2.0, -2.0, 0.0, 2.0, 2.0, 4.0, 4.0, 1.0, -1.0, 2.0, -2.0}));
  }

  template <ScalarType DTYPE>
  void test_round_execution_ints() {
    TensorFactory<DTYPE> tf;

    const std::vector<int32_t> sizes = {6};

    Tensor in = tf.make(sizes, /*data=*/{-1, 2, 0, 3, 0, -5});

    // Destination for the round.
    Tensor out = tf.zeros(sizes);

    // Run round.
    op_round_out(in, out);

    // Check that it matches the expected output.
    EXPECT_TENSOR_EQ(
        out,
        tf.make(
            sizes,
            /*data=*/
            {-1, 2, 0, 3, 0, -5}));
  }
};

TEST_F(OpRoundTest, FloatTensors) {
  test_round_execution_floats<ScalarType::Float>();
}

TEST_F(OpRoundTest, DoubleTensors) {
  test_round_execution_floats<ScalarType::Double>();
}

TEST_F(OpRoundTest, ByteTensors) {
  TensorFactory<ScalarType::Byte> tf;

  const std::vector<int32_t> sizes = {6};

  Tensor in = tf.make(sizes, /*data=*/{1, 2, 0, 3, 0, 5});

  // Destination for the round.
  Tensor out = tf.zeros(sizes);

  // Run round.
  op_round_out(in, out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_EQ(
      out,
      tf.make(
          sizes,
          /*data=*/
          {1, 2, 0, 3, 0, 5}));
}

TEST_F(OpRoundTest, CharTensors) {
  test_round_execution_ints<ScalarType::Char>();
}

TEST_F(OpRoundTest, ShortTensors) {
  test_round_execution_ints<ScalarType::Short>();
}

TEST_F(OpRoundTest, IntTensors) {
  test_round_execution_ints<ScalarType::Int>();
}

TEST_F(OpRoundTest, LongTensors) {
  test_round_execution_ints<ScalarType::Long>();
}

TEST_F(OpRoundTest, InfAndNanPreserved) {
  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> sizes = {7};

  Tensor in = tf.make(
      sizes,
      /*data=*/
      {1.7, 1.4, NAN, std::numeric_limits<float>::infinity(), 1.5, -1.5, 0});

  // Destination for the round.
  Tensor out = tf.zeros(sizes);

  // Run full round.
  op_round_out(in, out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_EQ(
      out,
      tf.make(
          sizes,
          /*data=*/
          {2.0,
           1.0,
           NAN,
           std::numeric_limits<float>::infinity(),
           2.0,
           -2.0,
           0.0}));
}

TEST_F(OpRoundTest, UnhandledDtypeDies) {
  // round() doesn't handle Bool.
  TensorFactory<ScalarType::Bool> tf;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor a = tf.make(sizes, /*data=*/{false, true, false, true});

  // Destination for the round.
  Tensor out = tf.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(context_, op_round_out(a, out));
}

/* %python
import torch
torch.manual_seed(0)
x = torch.rand(3, 2) * 10.0 - 5.0
res = torch.round(x)
op = "op_round_out"
dtype = "ScalarType::Float"
check = "EXPECT_TENSOR_EQ" */

TEST_F(OpRoundTest, DynamicShapeUpperBoundSameAsExpected) {
  /* %python
  out_args = "{3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {-0.03743410110473633,
       2.682218074798584,
       -4.115225791931152,
       -3.6796951293945312,
       -1.925771713256836,
       1.3407869338989258});
  Tensor expected = tf.make({3, 2}, {-0.0, 3.0, -4.0, -4.0, -2.0, 1.0});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_round_out(x, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpRoundTest, DynamicShapeUpperBoundLargerThanExpected) {
  /* %python
  out_args = "{10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {-0.03743410110473633,
       2.682218074798584,
       -4.115225791931152,
       -3.6796951293945312,
       -1.925771713256836,
       1.3407869338989258});
  Tensor expected = tf.make({3, 2}, {-0.0, 3.0, -4.0, -4.0, -2.0, 1.0});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_round_out(x, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpRoundTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape unbound not supported";
  /* %python
  out_args = "{1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {-0.03743410110473633,
       2.682218074798584,
       -4.115225791931152,
       -3.6796951293945312,
       -1.925771713256836,
       1.3407869338989258});
  Tensor expected = tf.make({3, 2}, {-0.0, 3.0, -4.0, -4.0, -2.0, 1.0});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  op_round_out(x, out);
  EXPECT_TENSOR_EQ(out, expected);
}
