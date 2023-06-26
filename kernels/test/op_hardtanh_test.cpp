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

TEST(OpHardTanhTest, SanityCheck) {
  TensorFactory<ScalarType::Float> tf;
  Tensor in = tf.ones({2, 2});
  Tensor out = tf.zeros({2, 2});

  exec_aten::RuntimeContext context{};
  Tensor ret = torch::executor::aten::hardtanh_outf(context, in, -2, 2, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf.ones({2, 2}));
}
