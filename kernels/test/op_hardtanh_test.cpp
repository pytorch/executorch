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

class OpHardTanhTest : public OperatorTest {
 protected:
  Tensor& op_hardtanh_out(
      const Tensor& self,
      const Scalar& min_val,
      const Scalar& max_val,
      Tensor& out) {
    return torch::executor::aten::hardtanh_outf(
        context_, self, min_val, max_val, out);
  }

  template <ScalarType DTYPE>
  void test_dtype() {
    TensorFactory<DTYPE> tf;

    Tensor in = tf.make({2, 3}, {0, 1, 2, 3, 4, 5});
    Scalar min_val = 1;
    Scalar max_val = 4;
    Tensor out = tf.zeros({2, 3});
    Tensor expected = tf.make({2, 3}, {1, 1, 2, 3, 4, 4});

    Tensor ret = op_hardtanh_out(in, min_val, max_val, out);

    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(out, expected);
  }
};

TEST_F(OpHardTanhTest, AllRealHBF16Input) {
#define TEST_KERNEL(INPUT_CTYPE, INPUT_DTYPE) \
  test_dtype<ScalarType::INPUT_DTYPE>();

  ET_FORALL_REALHBF16_TYPES(TEST_KERNEL);
#undef TEST_KERNEL
}

TEST_F(OpHardTanhTest, SanityCheck) {
  TensorFactory<ScalarType::Float> tf;
  Tensor in = tf.ones({2, 2});
  Tensor out = tf.zeros({2, 2});

  Tensor ret = op_hardtanh_out(in, -2, 2, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf.ones({2, 2}));
}
