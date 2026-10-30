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

class OpAvgPool1DOutTest : public OperatorTest {
 protected:
  Tensor& pool(
      const Tensor& in,
      std::initializer_list<int64_t> kernel,
      std::initializer_list<int64_t> stride,
      std::initializer_list<int64_t> padding,
      bool ceil_mode,
      bool include_pad,
      Tensor& out) {
    return torch::executor::aten::avg_pool1d_outf(
        context_,
        in,
        {kernel.begin(), kernel.size()},
        {stride.begin(), stride.size()},
        {padding.begin(), padding.size()},
        ceil_mode,
        include_pad,
        out);
  }
};

TEST_F(OpAvgPool1DOutTest, DefaultStride) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make({1, 2, 4}, {-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0});
  auto out = tf.zeros({1, 2, 2});
  pool(in, {2}, {}, {0}, false, true, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({1, 2, 2}, {-5.5, -3.5, -1.5, 0.5}));
}

TEST_F(OpAvgPool1DOutTest, Unbatched) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make({2, 3}, {-6.0, -5.0, -4.0, -3.0, -2.0, -1.0});
  auto out = tf.zeros({2, 2});
  pool(in, {2}, {1}, {0}, false, true, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({2, 2}, {-5.5, -4.5, -2.5, -1.5}));
}

TEST_F(OpAvgPool1DOutTest, IncludePadding) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make({1, 1, 3}, {-6.0, -5.0, -4.0});
  auto out = tf.zeros({1, 1, 2});
  pool(in, {3}, {2}, {1}, true, true, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({1, 1, 2}, {-3.6666667461395264, -3.0}));
}

TEST_F(OpAvgPool1DOutTest, ExcludePadding) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make({1, 1, 3}, {-6.0, -5.0, -4.0});
  auto out = tf.zeros({1, 1, 2});
  pool(in, {3}, {2}, {1}, true, false, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({1, 1, 2}, {-5.5, -4.5}));
}

TEST_F(OpAvgPool1DOutTest, CeilPartialWindow) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make({1, 1, 3}, {-6.0, -5.0, -4.0});
  auto out = tf.zeros({1, 1, 2});
  pool(in, {2}, {2}, {0}, true, true, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({1, 1, 2}, {-5.5, -4.0}));
}

TEST_F(OpAvgPool1DOutTest, CeilKernelLargerThanPaddedInputIncludePadding) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make({1, 1, 3}, {-6.0, -5.0, -4.0});
  auto out = tf.zeros({1, 1, 1});
  pool(in, {6}, {2}, {1}, true, true, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({1, 1, 1}, {-3.0}));
}

TEST_F(OpAvgPool1DOutTest, CeilKernelLargerThanPaddedInputExcludePadding) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make({1, 1, 3}, {-6.0, -5.0, -4.0});
  auto out = tf.zeros({1, 1, 1});
  pool(in, {6}, {2}, {1}, true, false, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({1, 1, 1}, {-5.0}));
}

TEST_F(OpAvgPool1DOutTest, CeilKernelLargerThanPaddedInputDefaultStride) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make({1, 1, 3}, {-6.0, -5.0, -4.0});
  auto out = tf.zeros({1, 1, 1});
  pool(in, {6}, {}, {1}, true, true, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({1, 1, 1}, {-3.0}));
}

TEST_F(OpAvgPool1DOutTest, FloorKernelLargerThanPaddedInput) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 3});
  auto out = tf.zeros({1, 1, 1});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {6}, {2}, {1}, false, true, out));
}

TEST_F(OpAvgPool1DOutTest, CeilKernelTooLarge) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 3});
  auto out = tf.zeros({1, 1, 1});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {7}, {2}, {1}, true, true, out));
}

TEST_F(OpAvgPool1DOutTest, CeilSkipsPaddingOnly) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make({1, 1, 2}, {-6.0, -5.0});
  auto out = tf.zeros({1, 1, 1});
  pool(in, {2}, {3}, {1}, true, true, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({1, 1, 1}, {-3.0}));
}

TEST_F(OpAvgPool1DOutTest, EmptyBatch) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make({0, 2, 3}, {});
  auto out = tf.zeros({0, 2, 2});
  pool(in, {2}, {1}, {0}, false, true, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({0, 2, 2}, {}));
}

TEST_F(OpAvgPool1DOutTest, KernelLargerThanInput) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.make({1, 1, 1}, {-6.0});
  auto out = tf.zeros({1, 1, 1});
  pool(in, {3}, {1}, {1}, false, true, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({1, 1, 1}, {-2.0}));
}

TEST_F(OpAvgPool1DOutTest, DoubleDtype) {
  TensorFactory<ScalarType::Double> tf;
  auto in = tf.full({1, 1, 3}, 4);
  auto out = tf.zeros({1, 1, 1});
  pool(in, {3}, {}, {0}, false, true, out);
  EXPECT_TENSOR_EQ(out, tf.full({1, 1, 1}, 4));
}

TEST_F(OpAvgPool1DOutTest, LongDtype) {
  TensorFactory<ScalarType::Long> tf;
  auto in = tf.full({1, 1, 3}, 4);
  auto out = tf.zeros({1, 1, 1});
  pool(in, {3}, {}, {0}, false, true, out);
  EXPECT_TENSOR_EQ(out, tf.full({1, 1, 1}, 4));
}

TEST_F(OpAvgPool1DOutTest, HalfDtype) {
  TensorFactory<ScalarType::Half> tf;
  auto in = tf.full({1, 1, 3}, 4);
  auto out = tf.zeros({1, 1, 1});
  pool(in, {3}, {}, {0}, false, true, out);
  EXPECT_TENSOR_EQ(out, tf.full({1, 1, 1}, 4));
}

TEST_F(OpAvgPool1DOutTest, BFloat16Dtype) {
  TensorFactory<ScalarType::BFloat16> tf;
  auto in = tf.full({1, 1, 3}, 4);
  auto out = tf.zeros({1, 1, 1});
  pool(in, {3}, {}, {0}, false, true, out);
  EXPECT_TENSOR_EQ(out, tf.full({1, 1, 1}, 4));
}

TEST_F(OpAvgPool1DOutTest, DynamicOutput) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.full({1, 1, 3}, 2);
  auto out = tf.zeros({1, 1, 3}, TensorShapeDynamism::DYNAMIC_BOUND);
  pool(in, {2}, {1}, {0}, false, true, out);
  EXPECT_TENSOR_EQ(out, tf.full({1, 1, 2}, 2));
}

TEST_F(OpAvgPool1DOutTest, ZeroKernel) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 3});
  auto out = tf.zeros({1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {0}, {1}, {0}, false, true, out));
}

TEST_F(OpAvgPool1DOutTest, NegativeKernel) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 3});
  auto out = tf.zeros({1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(
      context_, pool(in, {-1}, {1}, {0}, false, true, out));
}

TEST_F(OpAvgPool1DOutTest, ZeroStride) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 3});
  auto out = tf.zeros({1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, {0}, {0}, false, true, out));
}

TEST_F(OpAvgPool1DOutTest, NegativeStride) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 3});
  auto out = tf.zeros({1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(
      context_, pool(in, {2}, {-1}, {0}, false, true, out));
}

TEST_F(OpAvgPool1DOutTest, ExcessPadding) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 3});
  auto out = tf.zeros({1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, {1}, {2}, false, true, out));
}

TEST_F(OpAvgPool1DOutTest, NegativePadding) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 3});
  auto out = tf.zeros({1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(
      context_, pool(in, {2}, {1}, {-1}, false, true, out));
}

TEST_F(OpAvgPool1DOutTest, WrongKernelRank) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 3});
  auto out = tf.zeros({1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(
      context_, pool(in, {2, 2}, {1}, {0}, false, true, out));
}

TEST_F(OpAvgPool1DOutTest, WrongStrideRank) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 3});
  auto out = tf.zeros({1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(
      context_, pool(in, {2}, {1, 1}, {0}, false, true, out));
}

TEST_F(OpAvgPool1DOutTest, WrongPaddingRank) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 3});
  auto out = tf.zeros({1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(
      context_, pool(in, {2}, {1}, {0, 0}, false, true, out));
}

TEST_F(OpAvgPool1DOutTest, EmptyKernel) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 3});
  auto out = tf.zeros({1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {}, {1}, {0}, false, true, out));
}

TEST_F(OpAvgPool1DOutTest, WrongInputRank) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({3});
  auto out = tf.zeros({1, 1, 1});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, {1}, {0}, false, true, out));
}

TEST_F(OpAvgPool1DOutTest, EmptyChannels) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 0, 3});
  auto out = tf.zeros({1, 1, 1});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, {1}, {0}, false, true, out));
}

TEST_F(OpAvgPool1DOutTest, EmptySpatial) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 0});
  auto out = tf.zeros({1, 1, 1});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, {1}, {0}, false, true, out));
}

TEST_F(OpAvgPool1DOutTest, KernelTooLarge) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 1});
  auto out = tf.zeros({1, 1, 1});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, {1}, {0}, false, true, out));
}

TEST_F(OpAvgPool1DOutTest, MismatchedDtype) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Double> other;
  auto in = tf.ones({1, 1, 3});
  auto out = other.zeros({1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, {1}, {0}, false, true, out));
}

TEST_F(OpAvgPool1DOutTest, UnsupportedDtype) {
  TensorFactory<ScalarType::Int> tf;
  auto in = tf.ones({1, 1, 3});
  auto out = tf.zeros({1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, {1}, {0}, false, true, out));
}

TEST_F(OpAvgPool1DOutTest, IntegerTruncation) {
  TensorFactory<ScalarType::Long> tf;
  auto in = tf.make({1, 1, 2}, {-1, 0});
  auto out = tf.zeros({1, 1, 1});
  pool(in, {2}, {}, {0}, false, true, out);
  EXPECT_TENSOR_EQ(out, tf.zeros({1, 1, 1}));
}

#ifndef USE_ATEN_LIB
TEST_F(OpAvgPool1DOutTest, HalfAccumulatesInFloat) {
  TensorFactory<ScalarType::Half> tf;
  auto in = tf.full({1, 1, 3}, 60000);
  auto out = tf.zeros({1, 1, 1});
  pool(in, {3}, {}, {0}, false, true, out);
  EXPECT_TENSOR_EQ(out, tf.full({1, 1, 1}, 60000));
}

TEST_F(OpAvgPool1DOutTest, WrongOutputRank) {
  TensorFactory<ScalarType::Float> tf;
  auto in = tf.ones({1, 1, 3});
  auto out = tf.zeros({1});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, {1}, {0}, false, true, out));
}
#endif

#ifndef USE_ATEN_LIB
TEST_F(OpAvgPool1DOutTest, NonDefaultDimOrder) {
  TensorFactory<ScalarType::Float> tf;
  auto in =
      tf.make_with_dimorder({2, 2, 2}, std::vector<float>(8, 1), {0, 2, 1});
  auto out = tf.zeros({2, 2, 1});
  ET_EXPECT_KERNEL_FAILURE(context_, pool(in, {2}, {}, {0}, false, true, out));
}
#endif
