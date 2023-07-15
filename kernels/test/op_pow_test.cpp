// Copyright (c) Meta Platforms, Inc. and affiliates.

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

TEST(OpPowTest, TensorTensorSanityCheck) {
  TensorFactory<ScalarType::Byte> tf;
  Tensor self = tf.make({2, 2}, {2, 2, 2, 2});
  Tensor exp = tf.make({2, 1}, {4, 4});
  Tensor out = tf.make({2, 2}, {16, 16, 16, 16});

  exec_aten::RuntimeContext context{};
  Tensor ret = torch::executor::aten::pow_outf(context, self, exp, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {16, 16, 16, 16}));
}

TEST(OpPowTest, TensorTensorSanityCheck2) {
  TensorFactory<ScalarType::Float> tf1;
  TensorFactory<ScalarType::Int> tf2;
  TensorFactory<ScalarType::Double> tf3;

  Tensor self = tf1.make({2, 2}, {2, 3, 4, 5});
  Tensor exp = tf2.make({2, 1}, {2, 2});
  Tensor out = tf3.zeros({2, 2});

  exec_aten::RuntimeContext context{};
  Tensor ret = torch::executor::aten::pow_outf(context, self, exp, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf3.make({2, 2}, {4, 9, 16, 25}));
}

TEST(OpPowTest, TensorScalarSanityCheck) {
  TensorFactory<ScalarType::Byte> tf;
  Tensor self = tf.make({2, 2}, {2, 2, 2, 2});
  Tensor out = tf.make({2, 2}, {16, 16, 16, 16});

  exec_aten::RuntimeContext context{};
  Tensor ret = torch::executor::aten::pow_outf(context, self, 4, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {16, 16, 16, 16}));
}
