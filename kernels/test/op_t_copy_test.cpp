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
using exec_aten::ArrayRef;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor& _t_copy_out(const Tensor& self, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::t_copy_outf(context, self, out);
}

TEST(OpTCopyKernelTest, 1DTranspose) {
  TensorFactory<ScalarType::Int> tf;

  Tensor t_in = tf.make({4}, {1, 2, 3, 4});
  Tensor t_out = tf.make({4}, {0, 0, 0, 0});

  _t_copy_out(t_in, t_out);
  EXPECT_TENSOR_EQ(t_in, t_out);
}

TEST(OpTCopyKernelTest, 1DTransposeMismatchShapeDie) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched shapes";
  }
  TensorFactory<ScalarType::Int> tf;

  Tensor t_in = tf.make({4}, {1, 2, 3, 4});
  Tensor t_out = tf.make({2}, {0, 0});

  ET_EXPECT_KERNEL_FAILURE(_t_copy_out(t_in, t_out));
}

TEST(OpTCopyKernelTest, 2DTranspose) {
  TensorFactory<ScalarType::Int> tf;

  Tensor t_in = tf.make({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor t_out = tf.make({3, 2}, {0, 0, 0, 0, 0, 0});
  Tensor t_expected = tf.make({3, 2}, {1, 4, 2, 5, 3, 6});

  _t_copy_out(t_in, t_out);
  EXPECT_TENSOR_EQ(t_out, t_expected);
}

TEST(OpTCopyKernelTest, 2DTransposeMismatchShapeDie) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched shapes";
  }
  TensorFactory<ScalarType::Int> tf;

  Tensor t_in = tf.make({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor t_out = tf.make({2, 2}, {0, 0, 0, 0});

  ET_EXPECT_KERNEL_FAILURE(_t_copy_out(t_in, t_out));
}

TEST(OpTCopyKernelTest, 3DTransposeDie) {
  TensorFactory<ScalarType::Int> tf;

  Tensor t_in = tf.make({2, 3, 1}, {1, 2, 3, 4, 5, 6});
  Tensor t_out = tf.make({3, 2, 1}, {0, 0, 0, 0, 0, 0});

  ET_EXPECT_KERNEL_FAILURE(_t_copy_out(t_in, t_out));
}

/* %python
import torch
torch.manual_seed(0)
x = torch.rand(3, 2)
res = torch.t(x)
op = "_t_copy_out"
dtype = "ScalarType::Float"
check = "EXPECT_TENSOR_EQ" */

TEST(OpTCopyKernelTest, DynamicShapeUpperBoundSameAsExpected) {
  /* %python
  out_args = "{2, 3}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636});
  Tensor expected = tf.make(
      {2, 3},
      {0.49625658988952637,
       0.08847743272781372,
       0.30742281675338745,
       0.7682217955589294,
       0.13203048706054688,
       0.6340786814689636});

  Tensor out =
      tf.zeros({2, 3}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  _t_copy_out(x, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpTCopyKernelTest, DynamicShapeUpperBoundLargerThanExpected) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636});
  Tensor expected = tf.make(
      {2, 3},
      {0.49625658988952637,
       0.08847743272781372,
       0.30742281675338745,
       0.7682217955589294,
       0.13203048706054688,
       0.6340786814689636});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  _t_copy_out(x, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpTCopyKernelTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636});
  Tensor expected = tf.make(
      {2, 3},
      {0.49625658988952637,
       0.08847743272781372,
       0.30742281675338745,
       0.7682217955589294,
       0.13203048706054688,
       0.6340786814689636});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  _t_copy_out(x, out);
  EXPECT_TENSOR_EQ(out, expected);
}
