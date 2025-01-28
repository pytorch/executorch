/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/UnaryUfuncRealHBBF16ToFloatHBF16Test.h>

#include <gtest/gtest.h>

#include <cmath>

using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::SupportedFeatures;
using torch::executor::testing::TensorFactory;
class OpLogOutTest
    : public torch::executor::testing::UnaryUfuncRealHBBF16ToFloatHBF16Test {
 protected:
  Tensor& op_out(const Tensor& self, Tensor& out) override {
    return torch::executor::aten::log_outf(context_, self, out);
  }

  double op_reference(double x) const override {
    return std::log(x);
  }

  torch::executor::testing::SupportedFeatures* get_supported_features()
      const override;
};

IMPLEMENT_UNARY_UFUNC_REALHB_TO_FLOATH_TEST(OpLogOutTest)

TEST_F(OpLogOutTest, SimpleGeneratedCase) {
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
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

  Tensor out = tf.zeros({10, 10});
  Tensor ret = op_out(x, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpLogOutTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.6879220604896545,
       0.8289883136749268,
       0.7889447808265686,
       0.6339777112007141,
       0.8719115853309631,
       0.4185197353363037});
  Tensor expected_result = tf.make(
      {3, 2},
      {-0.37407973408699036,
       -0.18754921853542328,
       -0.23705895245075226,
       -0.4557414948940277,
       -0.1370672583580017,
       -0.8710312247276306});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_out(x, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}
