/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h>
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <gtest/gtest.h>
#include <initializer_list>
#include <vector>

using executorch::aten::ArrayRef;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::TensorShapeDynamism;
using executorch::runtime::testing::TensorFactory;

class OpAvgPool3DOutTest : public OperatorTest {
 protected:
  Tensor& pool(
      const Tensor& in,
      std::initializer_list<int64_t> kernel,
      std::initializer_list<int64_t> stride,
      std::initializer_list<int64_t> padding,
      bool ceil_mode,
      bool include_pad,
      Tensor& out,
      std::optional<int64_t> divisor = std::nullopt) {
    return torch::executor::aten::avg_pool3d_outf(
        context_,
        in,
        {kernel.begin(), kernel.size()},
        {stride.begin(), stride.size()},
        {padding.begin(), padding.size()},
        ceil_mode,
        include_pad,
        divisor,
        out);
  }
};

TEST_F(OpAvgPool3DOutTest, DefaultStride) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make(
      {1, 2, 4, 4, 4},
      {-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
       -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
       -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
       -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
       -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
       -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
       -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
       -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
       -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
       -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0});
  auto out = tf.zeros({1, 2, 2, 2, 2});
  pool(in, {2}, {}, {0}, false, true, out);
  EXPECT_TENSOR_CLOSE(
      out,
      tf.make(
          {1, 2, 2, 2, 2},
          {-2.0,
           0.0,
           1.125,
           -1.75,
           0.75,
           1.125,
           -1.0,
           1.0,
           -1.375,
           -1.0,
           1.75,
           -1.125,
           1.375,
           1.75,
           -2.0,
           0.0}));
}

TEST_F(OpAvgPool3DOutTest, Unbatched) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make(
      {2, 3, 3, 3},
      {-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0,  1.0,  2.0,  3.0,  4.0,
       5.0,  6.0,  -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0,  1.0,  2.0,
       3.0,  4.0,  5.0,  6.0,  -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0,
       1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  -6.0, -5.0, -4.0, -3.0, -2.0,
       -1.0, 0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  -6.0, -5.0});
  auto out = tf.zeros({2, 2, 2, 2});
  pool(in, {2}, {1}, {0}, false, true, out);
  EXPECT_TENSOR_CLOSE(
      out,
      tf.make(
          {2, 2, 2, 2},
          {-1.125,
           -1.75,
           -1.375,
           -2.0,
           1.375,
           0.75,
           1.125,
           -1.125,
           -1.75,
           -0.75,
           -2.0,
           -1.0,
           0.75,
           1.75,
           -1.125,
           -1.75}));
}

TEST_F(OpAvgPool3DOutTest, IncludePadding) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make(
      {1, 1, 3, 3, 3}, {-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0,  1.0,  2.0,
                        3.0,  4.0,  5.0,  6.0,  -6.0, -5.0, -4.0, -3.0, -2.0,
                        -1.0, 0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  -6.0});
  auto out = tf.zeros({1, 1, 2, 2, 2});
  pool(in, {3}, {2}, {1}, true, true, out);
  EXPECT_TENSOR_CLOSE(
      out,
      tf.make(
          {1, 1, 2, 2, 2},
          {-0.3333333432674408,
           -0.5185185074806213,
           -0.40740740299224854,
           -0.5925925970077515,
           0.40740740299224854,
           0.2222222238779068,
           0.3333333432674408,
           -0.3333333432674408}));
}

TEST_F(OpAvgPool3DOutTest, ExcludePadding) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make(
      {1, 1, 3, 3, 3}, {-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0,  1.0,  2.0,
                        3.0,  4.0,  5.0,  6.0,  -6.0, -5.0, -4.0, -3.0, -2.0,
                        -1.0, 0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  -6.0});
  auto out = tf.zeros({1, 1, 2, 2, 2});
  pool(in, {3}, {2}, {1}, true, false, out);
  EXPECT_TENSOR_CLOSE(
      out,
      tf.make(
          {1, 1, 2, 2, 2},
          {-1.125, -1.75, -1.375, -2.0, 1.375, 0.75, 1.125, -1.125}));
}

TEST_F(OpAvgPool3DOutTest, CeilPartialWindow) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make(
      {1, 1, 3, 3, 3}, {-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0,  1.0,  2.0,
                        3.0,  4.0,  5.0,  6.0,  -6.0, -5.0, -4.0, -3.0, -2.0,
                        -1.0, 0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  -6.0});
  auto out = tf.zeros({1, 1, 2, 2, 2});
  pool(in, {2}, {2}, {0}, true, true, out);
  EXPECT_TENSOR_CLOSE(
      out,
      tf.make(
          {1, 1, 2, 2, 2}, {-1.125, -1.25, -1.5, 0.0, 1.0, 2.5, 5.5, -6.0}));
}

TEST_F(OpAvgPool3DOutTest, CeilSkipsPaddingOnly) {
  TensorFactory<ScalarType::Float> tf;
  auto in =
      tf.make({1, 1, 2, 2, 2}, {-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0});
  auto out = tf.zeros({1, 1, 1, 1, 1});
  pool(in, {2}, {3}, {1}, true, true, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({1, 1, 1, 1, 1}, {-0.75}));
}

TEST_F(OpAvgPool3DOutTest, EmptyBatch) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make({0, 2, 3, 3, 3}, {});
  auto out = tf.zeros({0, 2, 2, 2, 2});
  pool(in, {2}, {1}, {0}, false, true, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({0, 2, 2, 2, 2}, {}));
}

TEST_F(OpAvgPool3DOutTest, Anisotropic) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make(
      {1, 1, 3, 4, 3},
      {-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0,  1.0,  2.0, 3.0, 4.0, 5.0,
       6.0,  -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0,  1.0, 2.0, 3.0, 4.0,
       5.0,  6.0,  -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0});
  auto out = tf.zeros({1, 1, 2, 2, 2});
  pool(in, {2, 3, 2}, {1, 2, 1}, {0, 1, 0}, false, false, out);
  EXPECT_TENSOR_CLOSE(
      out,
      tf.make(
          {1, 1, 2, 2, 2},
          {-2.875, -3.5, 0.0, 1.0, -0.625, -2.875, -1.0, 0.0}));
}

TEST_F(OpAvgPool3DOutTest, DivisorOverride) {
  TensorFactory<ScalarType::Float> tf;
  auto in =
      tf.make({1, 1, 2, 2, 2}, {-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0});
  auto out = tf.zeros({1, 1, 1, 1, 1});
  pool(in, {2}, {}, {0}, false, true, out, 4);
  EXPECT_TENSOR_CLOSE(out, tf.make({1, 1, 1, 1, 1}, {-5.0}));
}

TEST_F(OpAvgPool3DOutTest, NegativeDivisor) {
  TensorFactory<ScalarType::Float> tf;
  auto in =
      tf.make({1, 1, 2, 2, 2}, {-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0});
  auto out = tf.zeros({1, 1, 1, 1, 1});
  pool(in, {2}, {}, {0}, false, true, out, -4);
  EXPECT_TENSOR_CLOSE(out, tf.make({1, 1, 1, 1, 1}, {5.0}));
}

TEST_F(OpAvgPool3DOutTest, DoubleDtype) {
  TensorFactory<ScalarType::Double> tf;
  auto in = tf.full({1, 1, 3, 3, 3}, 4);
  auto out = tf.zeros({1, 1, 1, 1, 1});
  pool(in, {3}, {}, {0}, false, true, out);
  EXPECT_TENSOR_EQ(out, tf.full({1, 1, 1, 1, 1}, 4));
}

TEST_F(OpAvgPool3DOutTest, LongDtype) {
  TensorFactory<ScalarType::Long> tf;
  auto in = tf.full({1, 1, 3, 3, 3}, 4);
  auto out = tf.zeros({1, 1, 1, 1, 1});
  pool(in, {3}, {}, {0}, false, true, out);
  EXPECT_TENSOR_EQ(out, tf.full({1, 1, 1, 1, 1}, 4));
}

#ifndef USE_ATEN_LIB
TEST_F(OpAvgPool3DOutTest, HalfDtype) {
  TensorFactory<ScalarType::Half> tf;
  auto in = tf.full({1, 1, 3, 3, 3}, 4);
  auto out = tf.zeros({1, 1, 1, 1, 1});
  pool(in, {3}, {}, {0}, false, true, out);
  EXPECT_TENSOR_EQ(out, tf.full({1, 1, 1, 1, 1}, 4));
}

#endif
#ifndef USE_ATEN_LIB
TEST_F(OpAvgPool3DOutTest, BFloat16Dtype) {
  TensorFactory<ScalarType::BFloat16> tf;
  auto in = tf.full({1, 1, 3, 3, 3}, 4);
  auto out = tf.zeros({1, 1, 1, 1, 1});
  pool(in, {3}, {}, {0}, false, true, out);
  EXPECT_TENSOR_EQ(out, tf.full({1, 1, 1, 1, 1}, 4));
}

#endif
TEST_F(OpAvgPool3DOutTest, DynamicOutput) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.full({1, 1, 3, 3, 3}, 2);
  auto out = tf.zeros({1, 1, 3, 3, 3}, TensorShapeDynamism::DYNAMIC_BOUND);
  pool(in, {2}, {1}, {0}, false, true, out);
  EXPECT_TENSOR_EQ(out, tf.full({1, 1, 2, 2, 2}, 2));
}

TEST_F(OpAvgPool3DOutTest, ZeroKernel) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 3, 3, 3});
  auto out = tf.zeros({1, 1, 2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {0}, {1}, {0}, false, true, out));
}

TEST_F(OpAvgPool3DOutTest, NegativeKernel) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 3, 3, 3});
  auto out = tf.zeros({1, 1, 2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(
      context_, pool(in, {-1}, {1}, {0}, false, true, out));
}

TEST_F(OpAvgPool3DOutTest, ZeroStride) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 3, 3, 3});
  auto out = tf.zeros({1, 1, 2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, {0}, {0}, false, true, out));
}

TEST_F(OpAvgPool3DOutTest, NegativeStride) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 3, 3, 3});
  auto out = tf.zeros({1, 1, 2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(
      context_, pool(in, {2}, {-1}, {0}, false, true, out));
}

TEST_F(OpAvgPool3DOutTest, ExcessPadding) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 3, 3, 3});
  auto out = tf.zeros({1, 1, 2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, {1}, {2}, false, true, out));
}

TEST_F(OpAvgPool3DOutTest, NegativePadding) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 3, 3, 3});
  auto out = tf.zeros({1, 1, 2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(
      context_, pool(in, {2}, {1}, {-1}, false, true, out));
}

TEST_F(OpAvgPool3DOutTest, WrongKernelRank) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 3, 3, 3});
  auto out = tf.zeros({1, 1, 2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(
      context_, pool(in, {2, 2}, {1}, {0}, false, true, out));
}

TEST_F(OpAvgPool3DOutTest, WrongStrideRank) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 3, 3, 3});
  auto out = tf.zeros({1, 1, 2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(
      context_, pool(in, {2}, {1, 1}, {0}, false, true, out));
}

TEST_F(OpAvgPool3DOutTest, WrongPaddingRank) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 3, 3, 3});
  auto out = tf.zeros({1, 1, 2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(
      context_, pool(in, {2}, {1}, {0, 0}, false, true, out));
}

TEST_F(OpAvgPool3DOutTest, EmptyKernel) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 3, 3, 3});
  auto out = tf.zeros({1, 1, 2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {}, {1}, {0}, false, true, out));
}

TEST_F(OpAvgPool3DOutTest, WrongInputRank) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({3, 3, 3});
  auto out = tf.zeros({1, 1, 1, 1, 1});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, {1}, {0}, false, true, out));
}

TEST_F(OpAvgPool3DOutTest, EmptyChannels) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 0, 3, 3, 3});
  auto out = tf.zeros({1, 1, 1, 1, 1});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, {1}, {0}, false, true, out));
}

TEST_F(OpAvgPool3DOutTest, EmptySpatial) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 0, 0, 0});
  auto out = tf.zeros({1, 1, 1, 1, 1});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, {1}, {0}, false, true, out));
}

TEST_F(OpAvgPool3DOutTest, KernelTooLarge) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 1, 1, 1});
  auto out = tf.zeros({1, 1, 1, 1, 1});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, {1}, {0}, false, true, out));
}

TEST_F(OpAvgPool3DOutTest, MismatchedDtype) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Double> other;
  auto in = tf.ones({1, 1, 3, 3, 3});
  auto out = other.zeros({1, 1, 2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, {1}, {0}, false, true, out));
}

TEST_F(OpAvgPool3DOutTest, ZeroDivisor) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 2, 2, 2});
  auto out = tf.zeros({1, 1, 1, 1, 1});
  ET_EXPECT_KERNEL_FAILURE(
      context_, pool(in, {2}, {}, {0}, false, true, out, 0));
}

TEST_F(OpAvgPool3DOutTest, UnsupportedDtype) {
  TensorFactory<ScalarType::Int> tf;
  auto in = tf.ones({1, 1, 3, 3, 3});
  auto out = tf.zeros({1, 1, 2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, {1}, {0}, false, true, out));
}

TEST_F(OpAvgPool3DOutTest, IntegerTruncation) {
  TensorFactory<ScalarType::Long> tf;
  auto in = tf.make({1, 1, 2, 2, 2}, {-1, 0, 0, 0, 0, 0, 0, 0});
  auto out = tf.zeros({1, 1, 1, 1, 1});
  pool(in, {2}, {}, {0}, false, true, out);
  EXPECT_TENSOR_EQ(out, tf.zeros({1, 1, 1, 1, 1}));
}

#ifndef USE_ATEN_LIB
TEST_F(OpAvgPool3DOutTest, HalfAccumulatesInFloat) {
  TensorFactory<ScalarType::Half> tf;
  auto in = tf.full({1, 1, 3, 3, 3}, 60000);
  auto out = tf.zeros({1, 1, 1, 1, 1});
  pool(in, {3}, {}, {0}, false, true, out);
  EXPECT_TENSOR_EQ(out, tf.full({1, 1, 1, 1, 1}, 60000));
}

TEST_F(OpAvgPool3DOutTest, WrongOutputRank) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 3, 3, 3});
  auto out = tf.zeros({1});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, {1}, {0}, false, true, out));
}
#endif

#ifndef USE_ATEN_LIB
TEST_F(OpAvgPool3DOutTest, NonDefaultDimOrder) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make_with_dimorder(
      {2, 2, 2, 2, 2}, std::vector<float>(32, 1), {0, 2, 3, 4, 1});
  auto out = tf.zeros({2, 2, 1, 1, 1});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, {}, {0}, false, true, out));
}
#endif
