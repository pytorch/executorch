// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <executorch/core/kernel_types/kernel_types.h>
#include <executorch/core/kernel_types/testing/TensorFactory.h>
#include <executorch/core/kernel_types/testing/TensorUtil.h>
#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

TEST(OpLogicalXorTest, SanityCheck) {
  TensorFactory<ScalarType::Bool> tf;
  Tensor self = tf.make({2, 2}, {false, true, true, false});
  Tensor exp = tf.make({2, 1}, {true, false});
  Tensor out = tf.ones({2, 2});

  exec_aten::RuntimeContext context{};
  Tensor ret = torch::executor::aten::logical_xor_outf(context, self, exp, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {true, false, true, false}));
}
