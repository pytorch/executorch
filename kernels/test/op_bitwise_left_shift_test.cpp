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

class OpBitwiseLeftShiftTensorOutTest : public OperatorTest {
 protected:
  Tensor& op_bitwise_left_shift_tensor_out(
      const Tensor& self,
      const Tensor& other,
      Tensor& out) {
    return torch::executor::aten::bitwise_left_shift_outf(
        context_, self, other, out);
  }
};

class OpBitwiseLeftShiftScalarOutTest : public OperatorTest {
 protected:
  Tensor& op_bitwise_left_shift_scalar_out(
      const Tensor& self,
      const Scalar& other,
      Tensor& out) {
    return torch::executor::aten::bitwise_left_shift_outf(
        context_, self, other, out);
  }
};

TEST_F(OpBitwiseLeftShiftTensorOutTest, SmokeTestInt) {
  TensorFactory<ScalarType::Int> tf;

  // Test basic left shift: [1, 2, 4, 8] << [1, 2, 1, 2] = [2, 8, 8, 32]
  Tensor a = tf.make({2, 2}, {1, 2, 4, 8});
  Tensor b = tf.make({2, 2}, {1, 2, 1, 2});

  Tensor out = tf.zeros({2, 2});

  op_bitwise_left_shift_tensor_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {2, 8, 8, 32}));
}

TEST_F(OpBitwiseLeftShiftTensorOutTest, SmokeTestByte) {
  TensorFactory<ScalarType::Byte> tf;

  // Test with byte values: [1, 5, 10, 15] << [0, 1, 2, 3] = [1, 10, 40, 120]
  Tensor a = tf.make({2, 2}, {1, 5, 10, 15});
  Tensor b = tf.make({2, 2}, {0, 1, 2, 3});

  Tensor out = tf.zeros({2, 2});

  op_bitwise_left_shift_tensor_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {1, 10, 40, 120}));
}

TEST_F(OpBitwiseLeftShiftTensorOutTest, ZeroShift) {
  TensorFactory<ScalarType::Int> tf;

  // Shifting by 0 should return the original value
  Tensor a = tf.make({2, 2}, {5, 10, 15, 20});
  Tensor b = tf.zeros({2, 2});

  Tensor out = tf.zeros({2, 2});

  op_bitwise_left_shift_tensor_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {5, 10, 15, 20}));
}

TEST_F(OpBitwiseLeftShiftScalarOutTest, SmokeTestInt) {
  TensorFactory<ScalarType::Int> tf;

  // Test shifting by scalar: [1, 2, 3, 4] << 2 = [4, 8, 12, 16]
  Tensor a = tf.make({2, 2}, {1, 2, 3, 4});
  Scalar b = 2;

  Tensor out = tf.zeros({2, 2});

  op_bitwise_left_shift_scalar_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {4, 8, 12, 16}));
}

TEST_F(OpBitwiseLeftShiftScalarOutTest, ShiftByOne) {
  TensorFactory<ScalarType::Int> tf;

  // Shifting by 1 should double the value
  Tensor a = tf.make({2, 2}, {1, 5, 10, 100});
  Scalar b = 1;

  Tensor out = tf.zeros({2, 2});

  op_bitwise_left_shift_scalar_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {2, 10, 20, 200}));
}

TEST_F(OpBitwiseLeftShiftScalarOutTest, ShiftByZero) {
  TensorFactory<ScalarType::Int> tf;

  // Shifting by 0 should return the original value
  Tensor a = tf.make({2, 2}, {7, 14, 21, 28});
  Scalar b = 0;

  Tensor out = tf.zeros({2, 2});

  op_bitwise_left_shift_scalar_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {7, 14, 21, 28}));
}
