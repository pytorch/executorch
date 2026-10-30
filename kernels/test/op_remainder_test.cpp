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

#include <limits>

using namespace ::testing;
using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpRemainderOutTest : public OperatorTest {
 protected:
  Tensor& op_remainder_tensor_out(
      const Tensor& self,
      const Tensor& other,
      Tensor& out) {
    return torch::executor::aten::remainder_outf(context_, self, other, out);
  }

  Tensor& op_remainder_scalar_out(
      const Tensor& self,
      const Scalar& other,
      Tensor& out) {
    return torch::executor::aten::remainder_outf(context_, self, other, out);
  }
};

TEST_F(OpRemainderOutTest, SmokeTest) {
  TensorFactory<ScalarType::Long> tfDouble;
  TensorFactory<ScalarType::Long> tfLong;
  TensorFactory<ScalarType::Int> tfInt;

  Tensor self = tfLong.full({2, 2}, 46);
  Tensor other = tfInt.full({2, 2}, 4);
  Tensor out = tfDouble.zeros({2, 2});
  Tensor out_expected = tfDouble.full({2, 2}, 2.0);
  op_remainder_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST_F(OpRemainderOutTest, IntegerResultHasSignOfDivisor) {
  TensorFactory<ScalarType::Long> tf;

  Tensor self = tf.make({6}, {7, -7, 7, -7, 6, -6});
  Tensor other = tf.make({6}, {3, 3, -3, -3, 3, -3});
  Tensor out = tf.zeros({6});
  op_remainder_tensor_out(self, other, out);
  EXPECT_TENSOR_EQ(out, tf.make({6}, {1, 2, -2, -1, 0, 0}));

  Tensor scalar_out = tf.zeros({6});
  op_remainder_scalar_out(self, 3, scalar_out);
  EXPECT_TENSOR_EQ(scalar_out, tf.make({6}, {1, 2, 1, 2, 0, 0}));
}

TEST_F(OpRemainderOutTest, DoubleKeepsDoublePrecision) {
  TensorFactory<ScalarType::Double> tf;

  Tensor self = tf.make({2}, {123456789.123456789, -123456789.123456789});
  Tensor other = tf.make({2}, {1000.0, 1000.0});
  Tensor out = tf.zeros({2});
  op_remainder_tensor_out(self, other, out);
  // Rounding to float would be off by about 1.7e-5
  EXPECT_TENSOR_CLOSE_WITH_TOL(
      out, tf.make({2}, {789.123456789, 210.876543211}), 0, 1e-6);
}

TEST_F(OpRemainderOutTest, MostNegativeByMinusOneIsZero) {
  TensorFactory<ScalarType::Long> tf;

  const int64_t min = std::numeric_limits<int64_t>::min();
  Tensor self = tf.make({3}, {min, 7, -7});
  Tensor out = tf.zeros({3});
  op_remainder_tensor_out(self, tf.full({3}, -1), out);
  EXPECT_TENSOR_EQ(out, tf.zeros({3}));

  Tensor scalar_out = tf.full({3}, 1);
  op_remainder_scalar_out(self, -1, scalar_out);
  EXPECT_TENSOR_EQ(scalar_out, tf.zeros({3}));
}

TEST_F(OpRemainderOutTest, IntegerDivisionByZeroFails) {
  TensorFactory<ScalarType::Int> tf;

  Tensor self = tf.make({2}, {5, -3});
  Tensor out = tf.zeros({2});
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_remainder_tensor_out(self, tf.make({2}, {1, 0}), out));
  ET_EXPECT_KERNEL_FAILURE(context_, op_remainder_scalar_out(self, 0, out));
}

TEST_F(OpRemainderOutTest, ScalarIsCastToIntegerDtype) {
  TensorFactory<ScalarType::Char> tf;

  Tensor self = tf.make({3}, {5, -3, 100});
  Tensor out = tf.full({3}, 1);
  // 255 and 257 become -1 and 1 as int8.
  op_remainder_scalar_out(self, 255, out);
  EXPECT_TENSOR_EQ(out, tf.zeros({3}));
  out = tf.full({3}, 1);
  op_remainder_scalar_out(self, 257, out);
  EXPECT_TENSOR_EQ(out, tf.zeros({3}));
  // 256 becomes 0 as int8.
  ET_EXPECT_KERNEL_FAILURE(context_, op_remainder_scalar_out(self, 256, out));
}

TEST_F(OpRemainderOutTest, EmptyInputWithZeroScalarDivisor) {
  TensorFactory<ScalarType::Char> tf;

  Tensor self = tf.make({0}, {});
  Tensor out = tf.make({0}, {});
  op_remainder_scalar_out(self, 256, out);
  EXPECT_TENSOR_EQ(out, tf.make({0}, {}));
  op_remainder_scalar_out(self, 0, out);
  EXPECT_TENSOR_EQ(out, tf.make({0}, {}));
}
