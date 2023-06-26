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

Tensor& _floor_out(const Tensor& self, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::floor_outf(context, self, out);
}

TEST(OpFloorTest, SanityCheck) {
  TensorFactory<ScalarType::Float> tf;

  Tensor in = tf.make({1, 7}, {-3.0, -2.99, -1.01, 0.0, 1.01, 2.99, 3.0});
  Tensor out = tf.zeros({1, 7});
  Tensor expected = tf.make({1, 7}, {-3.0, -3.0, -2.0, 0.0, 1.0, 2.0, 3.0});

  Tensor ret = _floor_out(in, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, expected);
}
