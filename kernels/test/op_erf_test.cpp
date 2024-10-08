/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/UnaryUfuncRealHBBF16ToFloatHBF16Test.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpErfOutTest
    : public torch::executor::testing::UnaryUfuncRealHBBF16ToFloatHBF16Test {
 protected:
  Tensor& op_out(const Tensor& self, Tensor& out) override {
    return torch::executor::aten::erf_outf(context_, self, out);
  }

  double op_reference(double x) const override {
    return std::erf(x);
  }

  torch::executor::testing::SupportedFeatures* get_supported_features()
      const override;
};

TEST_F(OpErfOutTest, SanityCheck) {
  TensorFactory<ScalarType::Float> tf;

  Tensor in = tf.make({1, 7}, {-3.0, -2.99, -1.01, 0.0, 1.01, 2.99, 3.0});
  Tensor out = tf.zeros({1, 7});
  // clang-format off
  Tensor expected = tf.make({1, 7}, {-0.999978, -0.999976, -0.846811,  0.000000,  0.846811,  0.999976, 0.999978});
  // clang-format on

  Tensor ret = op_out(in, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_CLOSE(out, expected);
}

IMPLEMENT_UNARY_UFUNC_REALHB_TO_FLOATH_TEST(OpErfOutTest)
