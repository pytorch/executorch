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
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpPowTest : public OperatorTest {
 protected:
  Tensor&
  op_pow_scalar_out(const Scalar& self, const Tensor& exponent, Tensor& out) {
    return torch::executor::aten::pow_outf(context_, self, exponent, out);
  }

  Tensor& op_pow_tensor_scalar_out(
      const Tensor& self,
      const Scalar& exponent,
      Tensor& out) {
    return torch::executor::aten::pow_outf(context_, self, exponent, out);
  }

  Tensor& op_pow_tensor_tensor_out(
      const Tensor& self,
      const Tensor& exponent,
      Tensor& out) {
    return torch::executor::aten::pow_outf(context_, self, exponent, out);
  }
};

TEST_F(OpPowTest, TensorTensorSanityCheck) {
  TensorFactory<ScalarType::Byte> tf;
  Tensor self = tf.make({2, 2}, {2, 2, 2, 2});
  Tensor exp = tf.make({2, 1}, {4, 4});
  Tensor out = tf.make({2, 2}, {16, 16, 16, 16});

  Tensor ret = op_pow_tensor_tensor_out(self, exp, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {16, 16, 16, 16}));
}

TEST_F(OpPowTest, TensorTensorSanityCheck2) {
  TensorFactory<ScalarType::Float> tf1;
  TensorFactory<ScalarType::Int> tf2;
  TensorFactory<ScalarType::Double> tf3;

  Tensor self = tf1.make({2, 2}, {2, 3, 4, 5});
  Tensor exp = tf2.make({2, 1}, {2, 2});
  Tensor out = tf3.zeros({2, 2});

  Tensor ret = op_pow_tensor_tensor_out(self, exp, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf3.make({2, 2}, {4, 9, 16, 25}));
}

TEST_F(OpPowTest, TensorTensorHalfSupport) {
  TensorFactory<ScalarType::Half> tf;

  Tensor self = tf.make({2, 2}, {2.0, 3.0, 4.0, 5.0});
  Tensor exp = tf.make({2, 1}, {3.0, 2.0});
  Tensor out = tf.zeros({2, 2});

  Tensor ret = op_pow_tensor_tensor_out(self, exp, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {8.0, 27.0, 16.0, 25.0}));
}

TEST_F(OpPowTest, TensorScalarSanityCheck) {
  TensorFactory<ScalarType::Byte> tf;
  Tensor self = tf.make({2, 2}, {2, 2, 2, 2});
  Tensor out = tf.make({2, 2}, {16, 16, 16, 16});

  Tensor ret = op_pow_tensor_scalar_out(self, 4, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {16, 16, 16, 16}));
}

TEST_F(OpPowTest, TensorScalarHalfSupport) {
  TensorFactory<ScalarType::Half> tf;
  Tensor self = tf.make({2, 2}, {2.0, 2.0, 2.0, 2.0});
  Tensor out = tf.zeros({2, 2});

  Tensor ret = op_pow_tensor_scalar_out(self, 4, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {16.0, 16.0, 16.0, 16.0}));
}

TEST_F(OpPowTest, ScalarSanityCheck) {
  TensorFactory<ScalarType::Byte> tf;
  Tensor exp = tf.make({2, 2}, {2, 2, 2, 2});
  Tensor out = tf.make({2, 2}, {16, 16, 16, 16});

  Tensor ret = op_pow_scalar_out(4, exp, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {16, 16, 16, 16}));
}

TEST_F(OpPowTest, ScalarHalfSupport) {
  TensorFactory<ScalarType::Half> tf;
  Tensor exp = tf.make({2, 2}, {2, 2, 2, 2});
  Tensor out = tf.zeros({2, 2});

  Tensor ret = op_pow_scalar_out(4, exp, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {16, 16, 16, 16}));
}
