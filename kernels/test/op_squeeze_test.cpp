// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <executorch/core/kernel_types/kernel_types.h>
#include <executorch/core/kernel_types/testing/TensorFactory.h>
#include <executorch/core/kernel_types/testing/TensorUtil.h>
#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor& squeeze_copy_dim_out(const Tensor& self, int64_t dim, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::squeeze_copy_outf(context, self, dim, out);
}

namespace {

TEST(OpSqueezeKernelTest, DTypesMismatchDies) {
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Double> tf_d;
  Tensor t_in = tf_int.ones({2});
  Tensor t_out = tf_d.ones({2});
  int64_t dim = 0;

  ET_EXPECT_KERNEL_FAILURE(squeeze_copy_dim_out(t_in, dim, t_out));
}

TEST(OpSqueezeKernelTest, 0DTensorSqueeze) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t_in = tf.ones({});
  Tensor t_out = tf.zeros({});
  Tensor t_expected = tf.ones({});
  int64_t dim = 0;

  squeeze_copy_dim_out(t_in, dim, t_out);
  EXPECT_TENSOR_EQ(t_expected, t_out);
  EXPECT_TENSOR_DATA_EQ(t_expected, t_out);
}

TEST(OpSqueezeKernelTest, 0DTensorSqueezeInvalidDim1Dies) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t_in = tf.ones({});
  Tensor t_out = tf.ones({});
  int64_t dim = 1;

  ET_EXPECT_KERNEL_FAILURE(squeeze_copy_dim_out(t_in, dim, t_out));
}

TEST(OpSqueezeKernelTest, 1DTensorSqueezeTo0D) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t_in = tf.ones({1});
  Tensor t_out = tf.make({}, {99});
  Tensor t_expected = tf.make({}, {1});
  int64_t dim = 0;

  squeeze_copy_dim_out(t_in, dim, t_out);
  EXPECT_TENSOR_EQ(t_expected, t_out);
  EXPECT_TENSOR_DATA_EQ(t_expected, t_out);
}

TEST(OpSqueezeKernelTest, 2DTensorSqueezeUnchange) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t_in = tf.ones({2, 1});
  Tensor t_out = tf.make({2, 1}, {4, 3});
  Tensor t_expected = t_in;
  int64_t dim = 0;

  squeeze_copy_dim_out(t_in, dim, t_out);
  EXPECT_TENSOR_EQ(t_expected, t_out);
  EXPECT_TENSOR_DATA_EQ(t_expected, t_out);
}

TEST(OpSqueezeKernelTest, 2DTensorSqueezeTo1D) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t_in = tf.ones({2, 1});
  Tensor t_out = tf.make({2}, {4, 3});
  Tensor t_expected = tf.ones({2});
  int64_t dim = 1;

  squeeze_copy_dim_out(t_in, dim, t_out);
  EXPECT_TENSOR_EQ(t_expected, t_out);
  EXPECT_TENSOR_DATA_EQ(t_expected, t_out);
}

#ifndef USE_ATEN_LIB
TEST(OpSqueezeKernelTest, 2DTensorSqueezeDownwardDimResizeOut) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t_in = tf.ones({2, 1});
  Tensor t_out = tf.zeros(
      {4, 1},
      torch::executor::TensorShapeDynamism::DYNAMIC_BOUND); // okay to dwonward
                                                            // resize to (2, 1)
  Tensor t_expected = tf.ones({2, 1});
  int64_t dim = 0;

  squeeze_copy_dim_out(t_in, dim, t_out);
  EXPECT_TENSOR_EQ(t_expected, t_out);
  EXPECT_TENSOR_DATA_EQ(t_expected, t_out);
}

TEST(OpSqueezeKernelTest, 2DTensorSqueezeUpwardDimResizeOutDie) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t_in = tf.ones({2, 1});
  Tensor t_out = tf.zeros(
      {1, 1},
      torch::executor::TensorShapeDynamism::DYNAMIC_BOUND); // can NOT upward
                                                            // resize 0th dim
  Tensor t_expected = tf.ones({2, 1});
  int64_t dim = 0;

  ET_EXPECT_KERNEL_FAILURE(squeeze_copy_dim_out(t_in, dim, t_out));
}

TEST(OpSqueezeKernelTest, 2DTensorSqueezeRemoveADimResizeOutDie) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t_in = tf.ones({2, 1});
  Tensor t_out = tf.zeros(
      {2, 1, 3},
      torch::executor::TensorShapeDynamism::
          DYNAMIC_BOUND); // can NOT remove the 2nd dim via resizing
  Tensor t_expected = tf.ones({2, 1});
  int64_t dim = 0;

  ET_EXPECT_KERNEL_FAILURE(squeeze_copy_dim_out(t_in, dim, t_out));
}

TEST(OpSqueezeKernelTest, 2DTensorSqueezeAddDimsResizeOutDie) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t_in = tf.ones({2, 1});
  Tensor t_out = tf.zeros(
      {2},
      torch::executor::TensorShapeDynamism::
          DYNAMIC_BOUND); // can NOT add dim(s) via resizing
  Tensor t_expected = tf.ones({2, 1});
  int64_t dim = 0;

  ET_EXPECT_KERNEL_FAILURE(squeeze_copy_dim_out(t_in, dim, t_out));
}
#endif

TEST(OpSqueezeKernelTest, TensorSqueeze) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t_in = tf.make({3, 1, 2, 1}, {1, 2, 3, 4, 5, 6});
  Tensor t_out = tf.zeros({3, 2, 1});
  Tensor t_expected = tf.make({3, 2, 1}, {1, 2, 3, 4, 5, 6});
  int64_t dim = 1;

  squeeze_copy_dim_out(t_in, dim, t_out);
  EXPECT_TENSOR_EQ(t_expected, t_out);
  EXPECT_TENSOR_DATA_EQ(t_expected, t_out);
}

TEST(OpSqueezeKernelTest, TensorSqueezeNegativeDim) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t_in = tf.make({3, 1, 2, 1}, {1, 2, 3, 4, 5, 6});
  Tensor t_out = tf.zeros({3, 2, 1});
  Tensor t_expected = tf.make({3, 2, 1}, {1, 2, 3, 4, 5, 6});
  int64_t dim = -3;

  squeeze_copy_dim_out(t_in, dim, t_out);
  EXPECT_TENSOR_EQ(t_expected, t_out);
  EXPECT_TENSOR_DATA_EQ(t_expected, t_out);
}

TEST(OpSqueezeKernelTest, TensorSqueezeInvaidDim) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t_in = tf.make({3, 1, 2, 1}, {1, 2, 3, 4, 5, 6});
  Tensor t_out = tf.zeros({3, 2, 1});
  Tensor t_expected = tf.make({3, 2, 1}, {1, 2, 3, 4, 5, 6});
  std::vector<int64_t> invalid_dims = {t_in.dim(), -t_in.dim() - 1};

  for (const auto dim : invalid_dims) {
    ET_EXPECT_KERNEL_FAILURE(squeeze_copy_dim_out(t_in, dim, t_out));
  }
}

/* %python
import torch
torch.manual_seed(0)
x = torch.rand(2, 1, 4)
res = torch.squeeze(x, 1)
op = "squeeze_copy_dim_out"
opt_extra_params = "1,"
dtype = "ScalarType::Float"
check = "EXPECT_TENSOR_EQ" */

TEST(OpSqueezeKernelTest, DynamicShapeUpperBoundSameAsExpected) {
  /* %python
  out_args = "{2, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {2, 1, 4},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636,
       0.4900934100151062,
       0.8964447379112244});
  Tensor expected = tf.make(
      {2, 4},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636,
       0.4900934100151062,
       0.8964447379112244});

  Tensor out =
      tf.zeros({2, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  squeeze_copy_dim_out(x, 1, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpSqueezeKernelTest, DynamicShapeUpperBoundLargerThanExpected) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{5, 5}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {2, 1, 4},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636,
       0.4900934100151062,
       0.8964447379112244});
  Tensor expected = tf.make(
      {2, 4},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636,
       0.4900934100151062,
       0.8964447379112244});

  Tensor out =
      tf.zeros({5, 5}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  squeeze_copy_dim_out(x, 1, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpSqueezeKernelTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {2, 1, 4},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636,
       0.4900934100151062,
       0.8964447379112244});
  Tensor expected = tf.make(
      {2, 4},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636,
       0.4900934100151062,
       0.8964447379112244});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  squeeze_copy_dim_out(x, 1, out);
  EXPECT_TENSOR_EQ(out, expected);
}

} // namespace
