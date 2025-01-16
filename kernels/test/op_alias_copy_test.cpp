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

class OpAliasCopyTest : public OperatorTest {
 protected:
  exec_aten::Tensor& op_alias_copy_out(
      const exec_aten::Tensor& self,
      exec_aten::Tensor& out) {
    return torch::executor::aten::alias_copy_outf(context_, self, out);
  }
};

TEST_F(OpAliasCopyTest, SmokeTest) {
  TensorFactory<ScalarType::Float> tf;

  Tensor a = tf.make({2, 2}, {2, 3, 2, 5});
  Tensor out = tf.zeros({2, 2});

  op_alias_copy_out(a, out);
  EXPECT_TENSOR_EQ(a, out);
}
