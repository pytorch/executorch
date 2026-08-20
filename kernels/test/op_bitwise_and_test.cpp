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
using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpBitwiseAndTensorOutTest : public OperatorTest {
 protected:
  Tensor& op_bitwise_and_tensor_out(
      const Tensor& self,
      const Tensor& other,
      Tensor& out) {
    return torch::executor::aten::bitwise_and_outf(context_, self, other, out);
  }
};

class OpBitwiseAndScalarOutTest : public OperatorTest {
 protected:
  Tensor& op_bitwise_and_scalar_out(
      const Tensor& self,
      const Scalar& other,
      Tensor& out) {
    return torch::executor::aten::bitwise_and_outf(context_, self, other, out);
  }
};

TEST_F(OpBitwiseAndTensorOutTest, SmokeTestInt) {
  TensorFactory<ScalarType::Int> tf;

  Tensor a = tf.make({2, 2}, {2, 3, 2, 5});
  Tensor b = tf.make({2, 2}, {1, 6, 2, 3});

  Tensor out = tf.zeros({2, 2});

  op_bitwise_and_tensor_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {0, 2, 2, 1}));
}

TEST_F(OpBitwiseAndTensorOutTest, SmokeTestBool) {
  TensorFactory<ScalarType::Bool> tf;

  Tensor a = tf.make({2, 2}, {true, false, true, false});
  Tensor b = tf.make({2, 2}, {true, true, false, false});

  Tensor out = tf.zeros({2, 2});

  op_bitwise_and_tensor_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {true, false, false, false}));
}

TEST_F(OpBitwiseAndTensorOutTest, SmokeTestMixed) {
  TensorFactory<ScalarType::Int> tfInt;
  TensorFactory<ScalarType::Bool> tfBool;

  Tensor a = tfInt.make({2, 2}, {2, 3, 2, 5});
  Tensor b = tfBool.make({2, 2}, {true, true, false, false});

  Tensor out = tfInt.zeros({2, 2});

  op_bitwise_and_tensor_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tfInt.make({2, 2}, {0, 1, 0, 0}));
}

TEST_F(OpBitwiseAndScalarOutTest, SmokeTestInt) {
  TensorFactory<ScalarType::Int> tf;

  Tensor a = tf.make({2, 2}, {2, 3, 2, 5});
  Scalar b = 6;

  Tensor out = tf.zeros({2, 2});

  op_bitwise_and_scalar_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {2, 2, 2, 4}));
}

TEST_F(OpBitwiseAndScalarOutTest, SmokeTestBool) {
  TensorFactory<ScalarType::Bool> tf;

  Tensor a = tf.make({2, 2}, {true, false, true, false});
  Scalar b = true;

  Tensor out = tf.zeros({2, 2});

  op_bitwise_and_scalar_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {true, false, true, false}));
}
