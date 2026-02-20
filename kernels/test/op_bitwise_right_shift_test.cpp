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

class OpBitwiseRightShiftTensorOutTest : public OperatorTest {
 protected:
  Tensor& op_bitwise_right_shift_tensor_out(
      const Tensor& self,
      const Tensor& other,
      Tensor& out) {
    return torch::executor::aten::bitwise_right_shift_outf(
        context_, self, other, out);
  }
};

class OpBitwiseRightShiftScalarOutTest : public OperatorTest {
 protected:
  Tensor& op_bitwise_right_shift_scalar_out(
      const Tensor& self,
      const Scalar& other,
      Tensor& out) {
    return torch::executor::aten::bitwise_right_shift_outf(
        context_, self, other, out);
  }
};

TEST_F(OpBitwiseRightShiftTensorOutTest, SmokeTestInt) {
  TensorFactory<ScalarType::Int> tf;

  // Test basic right shift: [8, 16, 32, 64] >> [1, 2, 1, 3] = [4, 4, 16, 8]
  Tensor a = tf.make({2, 2}, {8, 16, 32, 64});
  Tensor b = tf.make({2, 2}, {1, 2, 1, 3});

  Tensor out = tf.zeros({2, 2});

  op_bitwise_right_shift_tensor_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {4, 4, 16, 8}));
}

TEST_F(OpBitwiseRightShiftTensorOutTest, SmokeTestByte) {
  TensorFactory<ScalarType::Byte> tf;

  // Test with byte values: [128, 64, 32, 16] >> [1, 1, 2, 3] = [64, 32, 8, 2]
  Tensor a = tf.make({2, 2}, {128, 64, 32, 16});
  Tensor b = tf.make({2, 2}, {1, 1, 2, 3});

  Tensor out = tf.zeros({2, 2});

  op_bitwise_right_shift_tensor_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {64, 32, 8, 2}));
}

TEST_F(OpBitwiseRightShiftTensorOutTest, ZeroShift) {
  TensorFactory<ScalarType::Int> tf;

  // Shifting by 0 should return the original value
  Tensor a = tf.make({2, 2}, {5, 10, 15, 20});
  Tensor b = tf.zeros({2, 2});

  Tensor out = tf.zeros({2, 2});

  op_bitwise_right_shift_tensor_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {5, 10, 15, 20}));
}

TEST_F(OpBitwiseRightShiftScalarOutTest, SmokeTestInt) {
  TensorFactory<ScalarType::Int> tf;

  // Test shifting by scalar: [16, 32, 48, 64] >> 2 = [4, 8, 12, 16]
  Tensor a = tf.make({2, 2}, {16, 32, 48, 64});
  Scalar b = 2;

  Tensor out = tf.zeros({2, 2});

  op_bitwise_right_shift_scalar_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {4, 8, 12, 16}));
}

TEST_F(OpBitwiseRightShiftScalarOutTest, ShiftByOne) {
  TensorFactory<ScalarType::Int> tf;

  // Shifting by 1 should halve the value (integer division)
  Tensor a = tf.make({2, 2}, {100, 50, 20, 10});
  Scalar b = 1;

  Tensor out = tf.zeros({2, 2});

  op_bitwise_right_shift_scalar_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {50, 25, 10, 5}));
}

TEST_F(OpBitwiseRightShiftScalarOutTest, ShiftByZero) {
  TensorFactory<ScalarType::Int> tf;

  // Shifting by 0 should return the original value
  Tensor a = tf.make({2, 2}, {7, 14, 21, 28});
  Scalar b = 0;

  Tensor out = tf.zeros({2, 2});

  op_bitwise_right_shift_scalar_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {7, 14, 21, 28}));
}
