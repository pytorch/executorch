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

class OpIsNanTest : public OperatorTest {
 protected:
  Tensor& op_isnan_out(const Tensor& self, Tensor& out) {
    return torch::executor::aten::isnan_outf(context_, self, out);
  }
};

TEST_F(OpIsNanTest, SanityCheckFloat) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Bool> tfb;

  Tensor in = tf.make(
      {1, 5}, {-1.0, 0.0, 1.0, NAN, std::numeric_limits<float>::infinity()});
  Tensor out = tfb.zeros({1, 5});
  Tensor expected = tfb.make({1, 5}, {false, false, false, true, false});

  Tensor ret = op_isnan_out(in, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpIsNanTest, SanityCheckHalf) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "Test Half support only for ExecuTorch mode";
  }
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Bool> tfb;

  Tensor in = tf.make(
      {1, 5}, {-1.0, 0.0, 1.0, NAN, std::numeric_limits<float>::infinity()});
  Tensor out = tfb.zeros({1, 5});
  Tensor expected = tfb.make({1, 5}, {false, false, false, true, false});

  Tensor ret = op_isnan_out(in, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpIsNanTest, SanityCheckByte) {
  TensorFactory<ScalarType::Byte> tf;
  TensorFactory<ScalarType::Bool> tfb;

  Tensor in = tf.make({1, 5}, {1, 2, 3, 4, 5});
  Tensor out = tfb.zeros({1, 5});
  Tensor expected = tfb.make({1, 5}, {false, false, false, false, false});

  Tensor ret = op_isnan_out(in, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpIsNanTest, SanityCheckBool) {
  TensorFactory<ScalarType::Bool> tfb;

  Tensor in = tfb.make({1, 5}, {true, false, true, true, false});
  Tensor out = tfb.zeros({1, 5});
  Tensor expected = tfb.make({1, 5}, {false, false, false, false, false});

  Tensor ret = op_isnan_out(in, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpIsNanTest, SanityCheckOutDtype) {
  TensorFactory<ScalarType::Int> tf;

  Tensor in = tf.make({1, 5}, {1, 2, 3, 4, 5});
  Tensor out = tf.zeros({1, 5});

  ET_EXPECT_KERNEL_FAILURE(context_, op_isnan_out(in, out));
}
