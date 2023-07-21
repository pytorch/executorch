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

TEST(OpFmodTest, TensorTensorSanityCheck) {
  TensorFactory<ScalarType::Float> tf;
  Tensor self = tf.make({2, 2}, {2, 3, 4, 5});
  Tensor modulo = tf.make({2, 1}, {3, 2});
  Tensor out = tf.zeros({2, 2});

  exec_aten::RuntimeContext context{};
  Tensor ret = torch::executor::aten::fmod_outf(context, self, modulo, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {2, 0, 0, 1}));
}

TEST(OpFmodTest, TensorScalarSanityCheck) {
  TensorFactory<ScalarType::Byte> tf;
  Tensor self = tf.make({2, 2}, {2, 3, 4, 5});
  Tensor out = tf.make({2, 2}, {2, 0, 1, 2});

  exec_aten::RuntimeContext context{};
  Tensor ret = torch::executor::aten::fmod_outf(context, self, 3, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {2, 0, 1, 2}));
}
