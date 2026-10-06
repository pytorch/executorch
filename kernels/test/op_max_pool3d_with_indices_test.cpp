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

#include <numeric>
#include <vector>

using namespace ::testing;
using executorch::aten::ArrayRef;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpMaxPool3dWithIndicesOutTest : public OperatorTest {
 protected:
  std::tuple<Tensor&, Tensor&> op_max_pool3d_with_indices_out(
      const Tensor& self,
      ArrayRef<int64_t> kernel_size,
      ArrayRef<int64_t> stride,
      ArrayRef<int64_t> padding,
      ArrayRef<int64_t> dilation,
      bool ceil_mode,
      Tensor& out,
      Tensor& indices) {
    return torch::executor::aten::max_pool3d_with_indices_outf(
        context_,
        self,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        out,
        indices);
  }

  // A ramp, so a maximum identifies its own position.
  static std::vector<float> ramp(size_t numel) {
    std::vector<float> values(numel);
    std::iota(values.begin(), values.end(), 0.0f);
    return values;
  }
};

// All references come from torch.nn.functional.max_pool3d(return_indices=True).

TEST_F(OpMaxPool3dWithIndicesOutTest, Basic) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor in = tf.make({1, 1, 3, 4, 4}, ramp(48));
  Tensor out = tf.zeros({1, 1, 1, 2, 2});
  Tensor indices = tf_long.zeros({1, 1, 1, 2, 2});

  const int64_t kernel[] = {2, 2, 2};
  const int64_t stride[] = {2, 2, 2};
  const int64_t padding[] = {0, 0, 0};
  const int64_t dilation[] = {1, 1, 1};
  op_max_pool3d_with_indices_out(
      in, kernel, stride, padding, dilation, false, out, indices);

  EXPECT_TENSOR_EQ(out, tf.make({1, 1, 1, 2, 2}, {21, 23, 29, 31}));
  EXPECT_TENSOR_EQ(indices, tf_long.make({1, 1, 1, 2, 2}, {21, 23, 29, 31}));
}

TEST_F(OpMaxPool3dWithIndicesOutTest, PaddingAndMultipleChannels) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor in = tf.make({1, 2, 3, 3, 3}, ramp(54));
  Tensor out = tf.zeros({1, 2, 2, 2, 2});
  Tensor indices = tf_long.zeros({1, 2, 2, 2, 2});

  const int64_t kernel[] = {2, 2, 2};
  const int64_t stride[] = {2, 2, 2};
  const int64_t padding[] = {1, 1, 1};
  const int64_t dilation[] = {1, 1, 1};
  op_max_pool3d_with_indices_out(
      in, kernel, stride, padding, dilation, false, out, indices);

  EXPECT_TENSOR_EQ(
      out,
      tf.make(
          {1, 2, 2, 2, 2},
          {0,  2,  6,  8,  18, 20, 24, 26,
           27, 29, 33, 35, 45, 47, 51, 53}));
  // Indices are relative to each channel's own volume, so they repeat.
  EXPECT_TENSOR_EQ(
      indices,
      tf_long.make(
          {1, 2, 2, 2, 2},
          {0, 2, 6, 8, 18, 20, 24, 26, 0, 2, 6, 8, 18, 20, 24, 26}));
}

TEST_F(OpMaxPool3dWithIndicesOutTest, CeilMode) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor in = tf.make({1, 2, 3, 3, 3}, ramp(54));
  Tensor out = tf.zeros({1, 2, 2, 2, 2});
  Tensor indices = tf_long.zeros({1, 2, 2, 2, 2});

  const int64_t kernel[] = {2, 2, 2};
  const int64_t stride[] = {2, 2, 2};
  const int64_t padding[] = {0, 0, 0};
  const int64_t dilation[] = {1, 1, 1};
  op_max_pool3d_with_indices_out(
      in, kernel, stride, padding, dilation, true, out, indices);

  EXPECT_TENSOR_EQ(
      out,
      tf.make(
          {1, 2, 2, 2, 2},
          {13, 14, 16, 17, 22, 23, 25, 26,
           40, 41, 43, 44, 49, 50, 52, 53}));
  EXPECT_TENSOR_EQ(
      indices,
      tf_long.make(
          {1, 2, 2, 2, 2},
          {13, 14, 16, 17, 22, 23, 25, 26, 13, 14, 16, 17, 22, 23, 25, 26}));
}

TEST_F(OpMaxPool3dWithIndicesOutTest, UnbatchedInput) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor in = tf.make({1, 3, 3, 3}, ramp(27));
  Tensor out = tf.zeros({1, 2, 2, 2});
  Tensor indices = tf_long.zeros({1, 2, 2, 2});

  const int64_t kernel[] = {2, 2, 2};
  const int64_t stride[] = {1, 1, 1};
  const int64_t padding[] = {0, 0, 0};
  const int64_t dilation[] = {1, 1, 1};
  op_max_pool3d_with_indices_out(
      in, kernel, stride, padding, dilation, false, out, indices);

  EXPECT_TENSOR_EQ(
      out, tf.make({1, 2, 2, 2}, {13, 14, 16, 17, 22, 23, 25, 26}));
  EXPECT_TENSOR_EQ(
      indices, tf_long.make({1, 2, 2, 2}, {13, 14, 16, 17, 22, 23, 25, 26}));
}

TEST_F(OpMaxPool3dWithIndicesOutTest, MixedKernelStridePaddingDilation) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor in = tf.make({1, 1, 5, 5, 5}, ramp(125));
  Tensor out = tf.zeros({1, 1, 3, 5, 2});
  Tensor indices = tf_long.zeros({1, 1, 3, 5, 2});

  const int64_t kernel[] = {2, 3, 2};
  const int64_t stride[] = {2, 1, 2};
  const int64_t padding[] = {1, 1, 0};
  const int64_t dilation[] = {2, 1, 1};
  op_max_pool3d_with_indices_out(
      in, kernel, stride, padding, dilation, false, out, indices);

  const std::vector<float> expected = {
      31, 33, 36, 38, 41, 43, 46, 48, 46, 48, 81, 83, 86, 88, 91,
      93, 96, 98, 96, 98, 81, 83, 86, 88, 91, 93, 96, 98, 96, 98};
  EXPECT_TENSOR_EQ(out, tf.make({1, 1, 3, 5, 2}, expected));
  std::vector<int64_t> expected_indices(expected.begin(), expected.end());
  EXPECT_TENSOR_EQ(
      indices, tf_long.make({1, 1, 3, 5, 2}, expected_indices));
}
