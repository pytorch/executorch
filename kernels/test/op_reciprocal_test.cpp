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

class OpReciprocalTest : public OperatorTest {
 protected:
  Tensor& op_reciprocal_out(const Tensor& self, Tensor& out) {
    return torch::executor::aten::reciprocal_outf(context_, self, out);
  }
};

TEST_F(OpReciprocalTest, SanityCheck) {
  TensorFactory<ScalarType::Float> tf;

  Tensor in = tf.make({1, 7}, {-3.0, -2.99, -1.01, 0.0, 1.01, 2.99, 3.0});
  Tensor out = tf.zeros({1, 7});
  // clang-format off
  Tensor expected = tf.make({1, 7}, {-0.333333, -0.334448, -0.990099, std::numeric_limits<float>::infinity(), 0.990099, 0.334448, 0.333333});
  // clang-format on

  Tensor ret = op_reciprocal_out(in, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST_F(OpReciprocalTest, HandleBoolInput) {
  TensorFactory<ScalarType::Bool> tf_bool;
  TensorFactory<ScalarType::Float> tf_float;

  const std::vector<int32_t> sizes = {1, 2};

  Tensor a = tf_bool.make(sizes, /*data=*/{false, true});
  Tensor out = tf_float.zeros(sizes);
  Tensor res = tf_float.make(sizes, /*data=*/{INFINITY, 1.0});

  EXPECT_TENSOR_CLOSE(op_reciprocal_out(a, out), res);
}

TEST_F(OpReciprocalTest, HandleHalfInput) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "Test Half support only for ExecuTorch mode";
  }
  TensorFactory<ScalarType::Half> tf_half;

  const std::vector<int32_t> sizes = {1, 2};

  Tensor a = tf_half.make(sizes, /*data=*/{5.0, -2.0});
  Tensor out = tf_half.zeros(sizes);
  Tensor res = tf_half.make(sizes, /*data=*/{0.2, -0.5});

  EXPECT_TENSOR_CLOSE(op_reciprocal_out(a, out), res);
}
