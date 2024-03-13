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

class OpLeakyReluTest : public OperatorTest {
 protected:
  Tensor& op_leaky_relu_out(
      const Tensor& in,
      const Scalar& negative_slope,
      Tensor& out) {
    return torch::executor::aten::leaky_relu_outf(
        context_, in, negative_slope, out);
  }
};

TEST_F(OpLeakyReluTest, SanityCheck) {
  TensorFactory<ScalarType::Float> tf;
  Tensor in = tf.ones({2, 2});
  Tensor out = tf.zeros({2, 2});

  Tensor ret = op_leaky_relu_out(in, -0.01, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf.ones({2, 2}));
}
