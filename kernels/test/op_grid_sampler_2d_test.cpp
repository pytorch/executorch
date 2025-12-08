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

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::SupportedFeatures;
using torch::executor::testing::TensorFactory;

class OpGridSampler2dTest : public OperatorTest {
 protected:
  Tensor& op_grid_sampler_2d_out(
      const Tensor& input,
      const Tensor& grid,
      int64_t interpolation_mode,
      int64_t padding_mode,
      bool align_corners,
      Tensor& out) {
    return torch::executor::aten::grid_sampler_2d_outf(
        context_,
        input,
        grid,
        interpolation_mode,
        padding_mode,
        align_corners,
        out);
  }

  template <class CTYPE, executorch::aten::ScalarType DTYPE>
  std::enable_if_t<std::is_floating_point_v<CTYPE>, void>
  test_grid_sampler_2d_dtype() {
    TensorFactory<DTYPE> tf;

    // Simple test: 2x2 input, identity-like grid
    const auto input = tf.make({1, 1, 2, 2}, {1, 2, 3, 4});
    const auto grid = tf.make(
        {1, 2, 2, 2},
        {
            -0.5,
            -0.5, // Top-left quadrant
            0.5,
            -0.5, // Top-right quadrant
            -0.5,
            0.5, // Bottom-left quadrant
            0.5,
            0.5 // Bottom-right quadrant
        });
    auto out = tf.zeros({1, 1, 2, 2});

    op_grid_sampler_2d_out(
        input,
        grid,
        0, // bilinear
        0, // zeros padding
        false,
        out);

    // Output should be close to input for this nearly-identity grid
    EXPECT_TENSOR_CLOSE(out, input);
  }

  template <class CTYPE, executorch::aten::ScalarType DTYPE>
  std::enable_if_t<!std::is_floating_point_v<CTYPE>, void>
  test_grid_sampler_2d_dtype() {
    // not supported
    return;
  }
};

//
// Bilinear interpolation tests
//

TEST_F(OpGridSampler2dTest, BilinearSimple) {
  TensorFactory<ScalarType::Float> tf;

  // 2x2 input, sample at exact pixel centers
  const auto input = tf.make({1, 1, 2, 2}, {1.0, 2.0, 3.0, 4.0});

  // Grid: sample at pixel centers in normalized coords [-1, 1]
  // For 2x2 with align_corners=false:
  //   pixel (0,0) is at normalized (-0.5, -0.5)
  //   pixel (1,1) is at normalized (0.5, 0.5)
  const auto grid = tf.make(
      {1, 2, 2, 2},
      {
          -0.5,
          -0.5, // Sample pixel (0,0) -> 1.0
          0.5,
          -0.5, // Sample pixel (1,0) -> 2.0
          -0.5,
          0.5, // Sample pixel (0,1) -> 3.0
          0.5,
          0.5 // Sample pixel (1,1) -> 4.0
      });
  auto out = tf.zeros({1, 1, 2, 2});

  op_grid_sampler_2d_out(
      input,
      grid,
      0, // bilinear
      0, // zeros padding
      false,
      out);

  const auto expected = tf.make({1, 1, 2, 2}, {1.0, 2.0, 3.0, 4.0});
  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST_F(OpGridSampler2dTest, BilinearInterpolation) {
  TensorFactory<ScalarType::Float> tf;

  // 2x2 input
  const auto input = tf.make({1, 1, 2, 2}, {1.0, 2.0, 3.0, 4.0});

  // Sample at center of image (should be average of all pixels)
  const auto grid = tf.make({1, 1, 1, 2}, {0.0, 0.0});
  auto out = tf.zeros({1, 1, 1, 1});

  op_grid_sampler_2d_out(
      input,
      grid,
      0, // bilinear
      0, // zeros padding
      false,
      out);

  // Center should be close to 2.5 (average of 1,2,3,4)
  const auto expected = tf.make({1, 1, 1, 1}, {2.5});
  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST_F(OpGridSampler2dTest, BilinearAlignCorners) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.make({1, 1, 2, 2}, {1.0, 2.0, 3.0, 4.0});

  // With align_corners=true, corners map exactly to pixel centers
  const auto grid = tf.make(
      {1, 2, 2, 2},
      {
          -1.0,
          -1.0, // Top-left corner -> pixel (0,0) -> 1.0
          1.0,
          -1.0, // Top-right corner -> pixel (1,0) -> 2.0
          -1.0,
          1.0, // Bottom-left corner -> pixel (0,1) -> 3.0
          1.0,
          1.0 // Bottom-right corner -> pixel (1,1) -> 4.0
      });
  auto out = tf.zeros({1, 1, 2, 2});

  op_grid_sampler_2d_out(
      input,
      grid,
      0, // bilinear
      0, // zeros padding
      true, // align_corners
      out);

  const auto expected = tf.make({1, 1, 2, 2}, {1.0, 2.0, 3.0, 4.0});
  EXPECT_TENSOR_EQ(out, expected);
}

//
// Nearest neighbor tests
//

TEST_F(OpGridSampler2dTest, NearestSimple) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.make({1, 1, 2, 2}, {1.0, 2.0, 3.0, 4.0});

  // Sample near pixel centers (should snap to nearest pixel)
  const auto grid = tf.make(
      {1, 2, 2, 2},
      {
          -0.6,
          -0.6, // Near (0,0) -> 1.0
          0.4,
          -0.4, // Near (1,0) -> 2.0
          -0.3,
          0.3, // Near (0,1) -> 3.0
          0.6,
          0.6 // Near (1,1) -> 4.0
      });
  auto out = tf.zeros({1, 1, 2, 2});

  op_grid_sampler_2d_out(
      input,
      grid,
      1, // nearest
      0, // zeros padding
      false,
      out);

  const auto expected = tf.make({1, 1, 2, 2}, {1.0, 2.0, 3.0, 4.0});
  EXPECT_TENSOR_EQ(out, expected);
}

//
// Bicubic interpolation tests
//

TEST_F(OpGridSampler2dTest, BicubicSimple) {
  TensorFactory<ScalarType::Float> tf;

  // Larger input for bicubic (needs 4x4 neighborhood)
  const auto input = tf.make(
      {1, 1, 4, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});

  // Sample at center
  const auto grid = tf.make({1, 1, 1, 2}, {0.0, 0.0});
  auto out = tf.zeros({1, 1, 1, 1});

  op_grid_sampler_2d_out(
      input,
      grid,
      2, // bicubic
      0, // zeros padding
      false,
      out);

  // Bicubic at center should be close to 8.5 (average of middle pixels)
  // Note: The tolerance of 0.5 is intentionally large because the expected
  // value (8.5) is a rough estimate (average of the middle pixels), not the
  // exact bicubic interpolation result. Bicubic interpolation can produce
  // values that differ from this average due to its mathematical properties.
  const auto expected = tf.make({1, 1, 1, 1}, {8.5});
  EXPECT_TENSOR_CLOSE_WITH_TOL(out, expected, 0, 0.5);
}

//
// Padding mode tests
//

TEST_F(OpGridSampler2dTest, ZerosPaddingOutOfBounds) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.make({1, 1, 2, 2}, {1.0, 2.0, 3.0, 4.0});

  // Sample way outside the image bounds
  const auto grid = tf.make(
      {1, 2, 2, 2},
      {
          -2.0,
          -2.0, // Far outside
          2.0,
          2.0, // Far outside
          -0.5,
          -0.5, // Inside
          0.5,
          0.5 // Inside
      });
  auto out = tf.zeros({1, 1, 2, 2});

  op_grid_sampler_2d_out(
      input,
      grid,
      0, // bilinear
      0, // zeros padding
      false,
      out);

  // Out-of-bounds samples should be 0, in-bounds samples should match
  const auto expected = tf.make({1, 1, 2, 2}, {0.0, 0.0, 1.0, 4.0});
  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST_F(OpGridSampler2dTest, BorderPaddingOutOfBounds) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.make({1, 1, 2, 2}, {1.0, 2.0, 3.0, 4.0});

  // Sample outside bounds
  const auto grid = tf.make(
      {1, 1, 2, 2},
      {
          -2.0,
          -2.0, // Should clamp to top-left pixel -> 1.0
          2.0,
          2.0 // Should clamp to bottom-right pixel -> 4.0
      });
  auto out = tf.zeros({1, 1, 1, 2});

  op_grid_sampler_2d_out(
      input,
      grid,
      0, // bilinear
      1, // border padding
      false,
      out);

  const auto expected = tf.make({1, 1, 1, 2}, {1.0, 4.0});
  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST_F(OpGridSampler2dTest, ReflectionPadding) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.make({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});

  // Sample with reflection padding
  const auto grid = tf.make({1, 1, 1, 2}, {0.0, 0.0});
  auto out = tf.zeros({1, 1, 1, 1});

  op_grid_sampler_2d_out(
      input,
      grid,
      0, // bilinear
      2, // reflection padding
      false,
      out);

  // Center pixel should be 5
  const auto expected = tf.make({1, 1, 1, 1}, {5.0});
  EXPECT_TENSOR_CLOSE(out, expected);
}

//
// Multi-channel and batch tests
//

TEST_F(OpGridSampler2dTest, MultiChannel) {
  TensorFactory<ScalarType::Float> tf;

  // 2 channels
  const auto input = tf.make(
      {1, 2, 2, 2},
      {1,
       2, // Channel 0
       3,
       4,
       5,
       6, // Channel 1
       7,
       8});

  const auto grid = tf.make({1, 1, 1, 2}, {0.0, 0.0});
  auto out = tf.zeros({1, 2, 1, 1});

  op_grid_sampler_2d_out(
      input,
      grid,
      0, // bilinear
      0, // zeros padding
      false,
      out);

  // Each channel should average its 4 pixels
  const auto expected = tf.make({1, 2, 1, 1}, {2.5, 6.5});
  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST_F(OpGridSampler2dTest, MultiBatch) {
  TensorFactory<ScalarType::Float> tf;

  // 2 batches
  const auto input = tf.make(
      {2, 1, 2, 2},
      {1,
       2, // Batch 0
       3,
       4,
       5,
       6, // Batch 1
       7,
       8});

  const auto grid = tf.make(
      {2, 1, 1, 2},
      {
          0.0,
          0.0, // Batch 0 samples center
          0.0,
          0.0 // Batch 1 samples center
      });
  auto out = tf.zeros({2, 1, 1, 1});

  op_grid_sampler_2d_out(
      input,
      grid,
      0, // bilinear
      0, // zeros padding
      false,
      out);

  // Each batch averages its 4 pixels
  const auto expected = tf.make({2, 1, 1, 1}, {2.5, 6.5});
  EXPECT_TENSOR_CLOSE(out, expected);
}

//
// Dtype tests
//

TEST_F(OpGridSampler2dTest, DType) {
#define TEST_ENTRY(ctype, dtype) \
  test_grid_sampler_2d_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

//
// Error case tests
//

TEST_F(OpGridSampler2dTest, InvalidInputRankDies) {
  TensorFactory<ScalarType::Float> tf;

  // Input must be 4D
  const auto input = tf.ones({1, 2, 2});
  const auto grid = tf.make({1, 1, 1, 2}, {0.0, 0.0});
  auto out = tf.zeros({1, 1, 1, 1});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_grid_sampler_2d_out(input, grid, 0, 0, false, out));
}

TEST_F(OpGridSampler2dTest, InvalidGridRankDies) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.ones({1, 1, 2, 2});
  // Grid must be 4D
  const auto grid = tf.make({1, 1, 2}, {0.0, 0.0});
  auto out = tf.zeros({1, 1, 1, 1});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_grid_sampler_2d_out(input, grid, 0, 0, false, out));
}

TEST_F(OpGridSampler2dTest, GridLastDimMustBe2Dies) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.ones({1, 1, 2, 2});
  // Grid's last dimension must be 2 (x, y coordinates)
  const auto grid = tf.ones({1, 1, 1, 3});
  auto out = tf.zeros({1, 1, 1, 1});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_grid_sampler_2d_out(input, grid, 0, 0, false, out));
}

TEST_F(OpGridSampler2dTest, BatchSizeMismatchDies) {
  TensorFactory<ScalarType::Float> tf;

  // Batch size must match between input and grid
  const auto input = tf.ones({1, 1, 2, 2});
  const auto grid = tf.make({2, 1, 1, 2}, {0.0, 0.0, 0.0, 0.0});
  auto out = tf.zeros({1, 1, 1, 1});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_grid_sampler_2d_out(input, grid, 0, 0, false, out));
}

TEST_F(OpGridSampler2dTest, MismatchedDTypeDies) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_long;

  const auto input = tf.ones({1, 1, 2, 2});
  const auto grid = tf.make({1, 1, 1, 2}, {0.0, 0.0});
  // Output dtype must match input dtype
  auto out = tf_long.zeros({1, 1, 1, 1});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_grid_sampler_2d_out(input, grid, 0, 0, false, out));
}

TEST_F(OpGridSampler2dTest, GridDTypeMismatchDies) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Double> tf_double;

  const auto input = tf.ones({1, 1, 2, 2});
  // Grid dtype must match input dtype
  const auto grid = tf_double.make({1, 1, 1, 2}, {0.0, 0.0});
  auto out = tf.zeros({1, 1, 1, 1});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_grid_sampler_2d_out(input, grid, 0, 0, false, out));
}

TEST_F(OpGridSampler2dTest, InvalidInterpolationModeDies) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.ones({1, 1, 2, 2});
  const auto grid = tf.make({1, 1, 1, 2}, {0.0, 0.0});
  auto out = tf.zeros({1, 1, 1, 1});

  // Invalid interpolation mode (valid: 0=bilinear, 1=nearest, 2=bicubic)
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_grid_sampler_2d_out(input, grid, 3, 0, false, out));
}

TEST_F(OpGridSampler2dTest, InvalidPaddingModeDies) {
  TensorFactory<ScalarType::Float> tf;

  const auto input = tf.ones({1, 1, 2, 2});
  const auto grid = tf.make({1, 1, 1, 2}, {0.0, 0.0});
  auto out = tf.zeros({1, 1, 1, 1});

  // Invalid padding mode (valid: 0=zeros, 1=border, 2=reflection)
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_grid_sampler_2d_out(input, grid, 0, 3, false, out));
}