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

class OpSignTest : public OperatorTest {
 protected:
  Tensor& op_sign_out(const Tensor& self, Tensor& out) {
    return torch::executor::aten::sign_outf(context_, self, out);
  }

  template <ScalarType DTYPE>
  void test_dtype() {
    TensorFactory<DTYPE> tf;

    Tensor in = tf.make({2, 3}, {-3, -2, -1, 0, 1, 2});
    Tensor out = tf.zeros({2, 3});
    Tensor expected = tf.make({2, 3}, {-1, -1, -1, 0, 1, 1});

    Tensor ret = op_sign_out(in, out);

    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(out, expected);
  }

  template <>
  void test_dtype<ScalarType::Byte>() {
    TensorFactory<ScalarType::Byte> tf;

    Tensor in = tf.make({2, 3}, {253, 254, 255, 0, 1, 2});
    Tensor out = tf.zeros({2, 3});
    Tensor expected = tf.make({2, 3}, {1, 1, 1, 0, 1, 1});

    Tensor ret = op_sign_out(in, out);

    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(out, expected);
  }
};

TEST_F(OpSignTest, AllRealHBF16Input) {
#define TEST_KERNEL(INPUT_CTYPE, INPUT_DTYPE) \
  test_dtype<ScalarType::INPUT_DTYPE>();

  ET_FORALL_REALHBF16_TYPES(TEST_KERNEL);
#undef TEST_KERNEL
}

TEST_F(OpSignTest, ETSanityCheckFloat) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen returns 0 on NAN input";
  }
  TensorFactory<ScalarType::Float> tf;

  Tensor in = tf.make({1, 7}, {-INFINITY, -3., -1.5, 0., 1.5, NAN, INFINITY});
  Tensor out = tf.zeros({1, 7});
  Tensor expected = tf.make({1, 7}, {-1., -1., -1., 0., 1., NAN, 1.});

  Tensor ret = op_sign_out(in, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST_F(OpSignTest, ATenSanityCheckFloat) {
  if (!torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ET returns NAN on NAN input";
  }
  TensorFactory<ScalarType::Float> tf;

  Tensor in = tf.make({1, 7}, {-INFINITY, -3., -1.5, 0., 1.5, NAN, INFINITY});
  Tensor out = tf.zeros({1, 7});
  Tensor expected = tf.make({1, 7}, {-1., -1., -1., 0., 1., 0., 1.});

  Tensor ret = op_sign_out(in, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST_F(OpSignTest, SanityCheckBool) {
  TensorFactory<ScalarType::Bool> tf;

  Tensor in = tf.make({1, 6}, {false, true, false, false, true, true});
  Tensor out = tf.zeros({1, 6});
  // clang-format off
  Tensor expected = tf.make({1, 6}, {false, true, false, false, true, true});
  // clang-format on

  Tensor ret = op_sign_out(in, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_CLOSE(out, expected);
}
