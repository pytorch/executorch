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

Tensor& _reciprocal_out(const Tensor& self, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::reciprocal_outf(context, self, out);
}

TEST(OpReciprocalTest, SanityCheck) {
  TensorFactory<ScalarType::Float> tf;

  Tensor in = tf.make({1, 7}, {-3.0, -2.99, -1.01, 0.0, 1.01, 2.99, 3.0});
  Tensor out = tf.zeros({1, 7});
  // clang-format off
  Tensor expected = tf.make({1, 7}, {-0.333333, -0.334448, -0.990099, std::numeric_limits<float>::infinity(), 0.990099, 0.334448, 0.333333});
  // clang-format on

  Tensor ret = _reciprocal_out(in, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_CLOSE(out, expected);
}
