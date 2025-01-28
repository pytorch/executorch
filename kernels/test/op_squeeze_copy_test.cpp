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

class OpSqueezeTest : public OperatorTest {
 protected:
  Tensor&
  op_squeeze_copy_dim_out(const Tensor& self, int64_t dim, Tensor& out) {
    return torch::executor::aten::squeeze_copy_outf(context_, self, dim, out);
  }
};

class OpSqueezeCopyDimsOutTest : public OperatorTest {
 protected:
  Tensor& op_squeeze_copy_dims_out(
      const Tensor& self,
      exec_aten::ArrayRef<int64_t> dims,
      Tensor& out) {
    return torch::executor::aten::squeeze_copy_outf(context_, self, dims, out);
  }
};

namespace {

TEST_F(OpSqueezeTest, DTypesMismatchDies) {
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Double> tf_d;
  Tensor t_in = tf_int.ones({2});
  Tensor t_out = tf_d.ones({2});
  int64_t dim = 0;

  ET_EXPECT_KERNEL_FAILURE(context_, op_squeeze_copy_dim_out(t_in, dim, t_out));
}

TEST_F(OpSqueezeTest, 0DTensorSqueeze) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t_in = tf.ones({});
  Tensor t_out = tf.zeros({});
  Tensor t_expected = tf.ones({});
  int64_t dim = 0;

  op_squeeze_copy_dim_out(t_in, dim, t_out);
  EXPECT_TENSOR_EQ(t_expected, t_out);
  EXPECT_TENSOR_DATA_EQ(t_expected, t_out);
}

TEST_F(OpSqueezeTest, 0DTensorSqueezeInvalidDim1Dies) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t_in = tf.ones({});
  Tensor t_out = tf.ones({});
  int64_t dim = 1;

  ET_EXPECT_KERNEL_FAILURE(context_, op_squeeze_copy_dim_out(t_in, dim, t_out));
}

TEST_F(OpSqueezeTest, 1DTensorSqueezeTo0D) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t_in = tf.ones({1});
  Tensor t_out = tf.make({}, {99});
  Tensor t_expected = tf.make({}, {1});
  int64_t dim = 0;

  op_squeeze_copy_dim_out(t_in, dim, t_out);
  EXPECT_TENSOR_EQ(t_expected, t_out);
  EXPECT_TENSOR_DATA_EQ(t_expected, t_out);
}

TEST_F(OpSqueezeTest, 2DTensorSqueezeUnchange) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t_in = tf.ones({2, 1});
  Tensor t_out = tf.make({2, 1}, {4, 3});
  Tensor t_expected = t_in;
  int64_t dim = 0;

  op_squeeze_copy_dim_out(t_in, dim, t_out);
  EXPECT_TENSOR_EQ(t_expected, t_out);
  EXPECT_TENSOR_DATA_EQ(t_expected, t_out);
}

TEST_F(OpSqueezeTest, 2DTensorSqueezeTo1D) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t_in = tf.ones({2, 1});
  Tensor t_out = tf.make({2}, {4, 3});
  Tensor t_expected = tf.ones({2});
  int64_t dim = 1;

  op_squeeze_copy_dim_out(t_in, dim, t_out);
  EXPECT_TENSOR_EQ(t_expected, t_out);
  EXPECT_TENSOR_DATA_EQ(t_expected, t_out);
}

#ifndef USE_ATEN_LIB
TEST_F(OpSqueezeTest, 2DTensorSqueezeDownwardDimResizeOut) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t_in = tf.ones({2, 1});
  Tensor t_out = tf.zeros(
      {4, 1},
      torch::executor::TensorShapeDynamism::DYNAMIC_BOUND); // okay to dwonward
                                                            // resize to (2, 1)
  Tensor t_expected = tf.ones({2, 1});
  int64_t dim = 0;

  op_squeeze_copy_dim_out(t_in, dim, t_out);
  EXPECT_TENSOR_EQ(t_expected, t_out);
  EXPECT_TENSOR_DATA_EQ(t_expected, t_out);
}

TEST_F(OpSqueezeTest, 2DTensorSqueezeUpwardDimResizeOutDie) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t_in = tf.ones({2, 1});
  Tensor t_out = tf.zeros(
      {1, 1},
      torch::executor::TensorShapeDynamism::DYNAMIC_BOUND); // can NOT upward
                                                            // resize 0th dim
  Tensor t_expected = tf.ones({2, 1});
  int64_t dim = 0;

  ET_EXPECT_KERNEL_FAILURE(context_, op_squeeze_copy_dim_out(t_in, dim, t_out));
}

TEST_F(OpSqueezeTest, 2DTensorSqueezeRemoveADimResizeOutDie) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t_in = tf.ones({2, 1});
  Tensor t_out = tf.zeros(
      {2, 1, 3},
      torch::executor::TensorShapeDynamism::
          DYNAMIC_BOUND); // can NOT remove the 2nd dim via resizing
  Tensor t_expected = tf.ones({2, 1});
  int64_t dim = 0;

  ET_EXPECT_KERNEL_FAILURE(context_, op_squeeze_copy_dim_out(t_in, dim, t_out));
}

TEST_F(OpSqueezeTest, 2DTensorSqueezeAddDimsResizeOutDie) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t_in = tf.ones({2, 1});
  Tensor t_out = tf.zeros(
      {2},
      torch::executor::TensorShapeDynamism::
          DYNAMIC_BOUND); // can NOT add dim(s) via resizing
  Tensor t_expected = tf.ones({2, 1});
  int64_t dim = 0;

  ET_EXPECT_KERNEL_FAILURE(context_, op_squeeze_copy_dim_out(t_in, dim, t_out));
}
#endif

TEST_F(OpSqueezeTest, TensorSqueeze) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t_in = tf.make({3, 1, 2, 1}, {1, 2, 3, 4, 5, 6});
  Tensor t_out = tf.zeros({3, 2, 1});
  Tensor t_expected = tf.make({3, 2, 1}, {1, 2, 3, 4, 5, 6});
  int64_t dim = 1;

  op_squeeze_copy_dim_out(t_in, dim, t_out);
  EXPECT_TENSOR_EQ(t_expected, t_out);
  EXPECT_TENSOR_DATA_EQ(t_expected, t_out);
}

TEST_F(OpSqueezeTest, TensorSqueezeNegativeDim) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t_in = tf.make({3, 1, 2, 1}, {1, 2, 3, 4, 5, 6});
  Tensor t_out = tf.zeros({3, 2, 1});
  Tensor t_expected = tf.make({3, 2, 1}, {1, 2, 3, 4, 5, 6});
  int64_t dim = -3;

  op_squeeze_copy_dim_out(t_in, dim, t_out);
  EXPECT_TENSOR_EQ(t_expected, t_out);
  EXPECT_TENSOR_DATA_EQ(t_expected, t_out);
}

TEST_F(OpSqueezeTest, TensorSqueezeInvaidDim) {
  TensorFactory<ScalarType::Int> tf;
  Tensor t_in = tf.make({3, 1, 2, 1}, {1, 2, 3, 4, 5, 6});
  Tensor t_out = tf.zeros({3, 2, 1});
  Tensor t_expected = tf.make({3, 2, 1}, {1, 2, 3, 4, 5, 6});
  std::vector<int64_t> invalid_dims = {t_in.dim(), -t_in.dim() - 1};

  for (const auto dim : invalid_dims) {
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_squeeze_copy_dim_out(t_in, dim, t_out));
  }
}

/* %python
import torch
torch.manual_seed(0)
x = torch.rand(2, 1, 4)
res = torch.squeeze(x, 1)
op = "op_squeeze_copy_dim_out"
opt_extra_params = "1,"
dtype = "ScalarType::Float"
check = "EXPECT_TENSOR_EQ" */

TEST_F(OpSqueezeTest, DynamicShapeUpperBoundSameAsExpected) {
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
  op_squeeze_copy_dim_out(x, 1, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpSqueezeTest, DynamicShapeUpperBoundLargerThanExpected) {
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
  op_squeeze_copy_dim_out(x, 1, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpSqueezeTest, DynamicShapeUnbound) {
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
  op_squeeze_copy_dim_out(x, 1, out);
  EXPECT_TENSOR_EQ(out, expected);
}

} // namespace

TEST_F(OpSqueezeCopyDimsOutTest, SanityTest4D) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make(
      {1, 2, 1, 5},
      {-26.5,
       5.75,
       95.75,
       92.625,
       -97.25,
       65.5,
       -92.25,
       -67.625,
       54.75,
       27.125});
  ::std::vector<int64_t> dim_vec = {0, 2};
  exec_aten::ArrayRef<int64_t> dim =
      exec_aten::ArrayRef<int64_t>(dim_vec.data(), dim_vec.size());
  exec_aten::Tensor out = tfFloat.zeros({2, 5});
  exec_aten::Tensor out_expected = tfFloat.make(
      {2, 5},
      {-26.5,
       5.75,
       95.75,
       92.625,
       -97.25,
       65.5,
       -92.25,
       -67.625,
       54.75,
       27.125});
  op_squeeze_copy_dims_out(self, dim, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST_F(OpSqueezeCopyDimsOutTest, SanityCheck5D) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make(
      {1, 2, 1, 5, 4},
      {-73.5,  -67.625, -54.375, 51.625,  -11.125, -28.625, -40.75,  45.625,
       84.375, 65.625,  95.125,  -47.125, -21.25,  32.25,   -86.125, 55.875,
       -62.25, 47.125,  -71.875, 43.0,    47.875,  -73.375, 97.75,   69.25,
       64.125, -59.875, 59.75,   -52.25,  59.5,    44.875,  -51.25,  20.875,
       -67.0,  32.5,    -26.625, 83.75,   45.5,    85.5,    -92.875, 60.0});
  ::std::vector<int64_t> dim_vec = {0, 3, 2, 1};
  exec_aten::ArrayRef<int64_t> dim =
      exec_aten::ArrayRef<int64_t>(dim_vec.data(), dim_vec.size());
  exec_aten::Tensor out = tfFloat.zeros({2, 5, 4});
  exec_aten::Tensor out_expected = tfFloat.make(
      {2, 5, 4},
      {-73.5,  -67.625, -54.375, 51.625,  -11.125, -28.625, -40.75,  45.625,
       84.375, 65.625,  95.125,  -47.125, -21.25,  32.25,   -86.125, 55.875,
       -62.25, 47.125,  -71.875, 43.0,    47.875,  -73.375, 97.75,   69.25,
       64.125, -59.875, 59.75,   -52.25,  59.5,    44.875,  -51.25,  20.875,
       -67.0,  32.5,    -26.625, 83.75,   45.5,    85.5,    -92.875, 60.0});
  op_squeeze_copy_dims_out(self, dim, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST_F(OpSqueezeCopyDimsOutTest, SanityCheck5DUnchanged) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make(
      {1, 2, 1, 5, 4},
      {-0.375,  -40.125, 5.75,   21.25,   -34.875, -19.375, 15.75,   -60.75,
       -41.75,  53.125,  -76.0,  -64.25,  -84.5,   -37.25,  -39.125, 22.875,
       -69.0,   30.25,   -21.25, 85.5,    8.875,   41.625,  12.375,  -1.125,
       -14.875, 78.5,    43.0,   -78.625, -58.625, -58.375, 47.5,    -67.375,
       -82.375, 35.0,    83.25,  49.625,  -9.875,  -46.75,  17.875,  -68.375});
  ::std::vector<int64_t> dim_vec = {1, 4, 3};
  exec_aten::ArrayRef<int64_t> dim =
      exec_aten::ArrayRef<int64_t>(dim_vec.data(), dim_vec.size());
  exec_aten::Tensor out = tfFloat.zeros({1, 2, 1, 5, 4});
  exec_aten::Tensor out_expected = tfFloat.make(
      {1, 2, 1, 5, 4},
      {-0.375,  -40.125, 5.75,   21.25,   -34.875, -19.375, 15.75,   -60.75,
       -41.75,  53.125,  -76.0,  -64.25,  -84.5,   -37.25,  -39.125, 22.875,
       -69.0,   30.25,   -21.25, 85.5,    8.875,   41.625,  12.375,  -1.125,
       -14.875, 78.5,    43.0,   -78.625, -58.625, -58.375, 47.5,    -67.375,
       -82.375, 35.0,    83.25,  49.625,  -9.875,  -46.75,  17.875,  -68.375});
  op_squeeze_copy_dims_out(self, dim, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}
