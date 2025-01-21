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

  template <ScalarType DTYPE>
  void test_dtype() {
    TensorFactory<DTYPE> tf;

    Tensor in = tf.make({2, 3}, {-3, -2, -1, 0, 1, 2});
    Tensor out = tf.zeros({2, 3});
    Tensor expected = tf.make({2, 3}, {3, 2, 1, 0, -1, -2});

    Tensor ret = op_neg_out(in, out);

    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(out, expected);
  }

  template <>
  void test_dtype<ScalarType::Byte>() {
    TensorFactory<ScalarType::Byte> tf;

    Tensor in = tf.make({2, 3}, {253, 254, 255, 0, 1, 2});
    Tensor out = tf.zeros({2, 3});
    Tensor expected = tf.make({2, 3}, {3, 2, 1, 0, 255, 254});

    Tensor ret = op_neg_out(in, out);

    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(out, expected);
  }
};

TEST_F(OpNegTest, AllRealHBF16Input) {
#define TEST_KERNEL(INPUT_CTYPE, INPUT_DTYPE) \
  test_dtype<ScalarType::INPUT_DTYPE>();

  ET_FORALL_REALHBF16_TYPES(TEST_KERNEL);
#undef TEST_KERNEL
}

TEST_F(OpNegTest, SanityCheck) {
  TensorFactory<ScalarType::Float> tf;

  Tensor in = tf.make({1, 7}, {-3.0, -2.5, -1.01, 0.0, 1.01, 2.5, 3.0});
  Tensor out = tf.zeros({1, 7});
  Tensor expected = tf.make({1, 7}, {3.0, 2.5, 1.01, 0.0, -1.01, -2.5, -3.0});

  Tensor ret = op_neg_out(in, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, expected);
}
