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

class OpNegTest : public OperatorTest {
 protected:
  Tensor& op_neg_out(const Tensor& self, Tensor& out) {
    return torch::executor::aten::neg_outf(context_, self, out);
  }
};

TEST_F(OpNegTest, SanityCheck) {
  TensorFactory<ScalarType::Float> tf;

  Tensor in = tf.make({1, 7}, {-3.0, -2.5, -1.01, 0.0, 1.01, 2.5, 3.0});
  Tensor out = tf.zeros({1, 7});
  Tensor expected = tf.make({1, 7}, {3.0, 2.5, 1.01, 0.0, -1.01, -2.5, -3.0});

  Tensor ret = op_neg_out(in, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, expected);
}
