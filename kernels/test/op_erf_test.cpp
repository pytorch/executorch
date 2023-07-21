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

Tensor& _erf_out(const Tensor& self, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::erf_outf(context, self, out);
}

TEST(OpErfTest, SanityCheck) {
  TensorFactory<ScalarType::Float> tf;

  Tensor in = tf.make({1, 7}, {-3.0, -2.99, -1.01, 0.0, 1.01, 2.99, 3.0});
  Tensor out = tf.zeros({1, 7});
  // clang-format off
  Tensor expected = tf.make({1, 7}, {-0.999978, -0.999976, -0.846811,  0.000000,  0.846811,  0.999976, 0.999978});
  // clang-format on

  Tensor ret = _erf_out(in, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST(OpErfTest, HandleBoolInput) {
  TensorFactory<ScalarType::Bool> tf_bool;
  TensorFactory<ScalarType::Float> tf_float;

  const std::vector<int32_t> sizes = {1, 2};

  Tensor a = tf_bool.make(sizes, /*data=*/{false, true});
  Tensor out = tf_float.zeros(sizes);
  Tensor res = tf_float.make(sizes, /*data=*/{0.000000, 0.842701});

  EXPECT_TENSOR_CLOSE(_erf_out(a, out), res);
}
