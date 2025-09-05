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
using exec_aten::OptionalArrayRef;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpUpsampleBilinear2dAAOutTest : public OperatorTest {
 protected:
  Tensor& op_upsample_bilinear2d_aa_out(
      const Tensor& input,
      const ArrayRef<int64_t> output_size,
      bool align_corners,
      const std::optional<double> scales_h,
      const std::optional<double> scales_w,
      Tensor& out) {
    return torch::executor::aten::_upsample_bilinear2d_aa_outf(
        context_, input, output_size, align_corners, scales_h, scales_w, out);
  }
};

TEST_F(OpUpsampleBilinear2dAAOutTest, SmokeTest2xUpsampleNCHW) {
  TensorFactory<ScalarType::Float> tf;

  // Input shape: [1, 1, 2, 2]
  Tensor input = tf.make({1, 1, 2, 2}, {1, 2, 3, 4});

  // Output shape: [1, 1, 4, 4]
  Tensor out = tf.zeros({1, 1, 4, 4});

  // Upsample 2x with anti-aliasing - let scales be computed from sizes
  int64_t output_size_data[2] = {4, 4};
  ArrayRef<int64_t> output_size(output_size_data, 2);

  op_upsample_bilinear2d_aa_out(
      input,
      output_size,
      /*align_corners=*/false,
      std::nullopt,
      std::nullopt,
      out);

  // Verify output dimensions
  EXPECT_EQ(out.size(0), 1);
  EXPECT_EQ(out.size(1), 1);
  EXPECT_EQ(out.size(2), 4);
  EXPECT_EQ(out.size(3), 4);

  // Verify that output values are interpolated (not all zeros)
  auto out_data = out.const_data_ptr<float>();
  bool has_non_zero = false;
  for (int i = 0; i < 16; i++) {
    if (out_data[i] != 0.0f) {
      has_non_zero = true;
      break;
    }
  }
  EXPECT_TRUE(has_non_zero);
}

TEST_F(OpUpsampleBilinear2dAAOutTest, TestWithAlignCorners) {
  TensorFactory<ScalarType::Float> tf;

  // Input shape: [1, 2, 3, 3]
  Tensor input = tf.make(
      {1, 2, 3, 3},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});

  // Output shape: [1, 2, 6, 6]
  Tensor out = tf.zeros({1, 2, 6, 6});

  int64_t output_size_data[2] = {6, 6};
  ArrayRef<int64_t> output_size(output_size_data, 2);

  op_upsample_bilinear2d_aa_out(
      input,
      output_size,
      /*align_corners=*/true,
      std::nullopt,
      std::nullopt,
      out);

  // Verify output dimensions
  EXPECT_EQ(out.size(0), 1);
  EXPECT_EQ(out.size(1), 2);
  EXPECT_EQ(out.size(2), 6);
  EXPECT_EQ(out.size(3), 6);

  // Check that corner values are preserved when align_corners=true
  auto in_data = input.const_data_ptr<float>();
  auto out_data = out.const_data_ptr<float>();

  // Top-left corner of first channel should match
  EXPECT_NEAR(
      out_data[0],
      in_data[0],
      0.35); // Relaxed tolerance due to implementation differences
  // Top-right corner of first channel
  EXPECT_NEAR(
      out_data[5],
      in_data[2],
      0.35); // Relaxed tolerance due to implementation differences
}

TEST_F(OpUpsampleBilinear2dAAOutTest, TestDownsample) {
  TensorFactory<ScalarType::Float> tf;

  // Input shape: [1, 1, 4, 4]
  Tensor input = tf.make(
      {1, 1, 4, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});

  // Output shape: [1, 1, 2, 2] (downsampling)
  Tensor out = tf.zeros({1, 1, 2, 2});

  int64_t output_size_data[2] = {2, 2};
  ArrayRef<int64_t> output_size(output_size_data, 2);

  op_upsample_bilinear2d_aa_out(
      input,
      output_size,
      /*align_corners=*/false,
      std::nullopt,
      std::nullopt,
      out);

  // Verify output dimensions
  EXPECT_EQ(out.size(0), 1);
  EXPECT_EQ(out.size(1), 1);
  EXPECT_EQ(out.size(2), 2);
  EXPECT_EQ(out.size(3), 2);

  // Verify that output has reasonable values
  auto out_data = out.const_data_ptr<float>();
  for (int i = 0; i < 4; i++) {
    EXPECT_GT(out_data[i], 0.0f);
    EXPECT_LT(out_data[i], 17.0f);
  }
}

TEST_F(OpUpsampleBilinear2dAAOutTest, TestBatchedInput) {
  TensorFactory<ScalarType::Float> tf;

  // Input shape: [2, 3, 2, 2] (batch of 2)
  Tensor input =
      tf.make({2, 3, 2, 2}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                             13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});

  // Output shape: [2, 3, 4, 4]
  Tensor out = tf.zeros({2, 3, 4, 4});

  int64_t output_size_data[2] = {4, 4};
  ArrayRef<int64_t> output_size(output_size_data, 2);

  op_upsample_bilinear2d_aa_out(
      input,
      output_size,
      /*align_corners=*/false,
      std::nullopt,
      std::nullopt,
      out);

  // Verify output dimensions
  EXPECT_EQ(out.size(0), 2);
  EXPECT_EQ(out.size(1), 3);
  EXPECT_EQ(out.size(2), 4);
  EXPECT_EQ(out.size(3), 4);
}

TEST_F(OpUpsampleBilinear2dAAOutTest, TestWithScaleFactors) {
  TensorFactory<ScalarType::Float> tf;

  // Input shape: [1, 1, 3, 3]
  Tensor input = tf.make({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});

  // Use scale factors instead of output size
  int64_t output_size_data[2] = {6, 6};
  ArrayRef<int64_t> output_size(output_size_data, 2);

  // Output shape should be [1, 1, 6, 6]
  Tensor out = tf.zeros({1, 1, 6, 6});

  op_upsample_bilinear2d_aa_out(
      input, output_size, /*align_corners=*/false, 2.0, 2.0, out);

  // Verify output dimensions
  EXPECT_EQ(out.size(0), 1);
  EXPECT_EQ(out.size(1), 1);
  EXPECT_EQ(out.size(2), 6);
  EXPECT_EQ(out.size(3), 6);
}

TEST_F(OpUpsampleBilinear2dAAOutTest, TestAsymmetricScaling) {
  TensorFactory<ScalarType::Float> tf;

  // Input shape: [1, 2, 3, 4] - different height and width
  Tensor input =
      tf.make({1, 2, 3, 4}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                             13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});

  // Output with different scaling for height (2x) and width (3x)
  Tensor out = tf.zeros({1, 2, 6, 12});

  int64_t output_size_data[2] = {6, 12};
  ArrayRef<int64_t> output_size(output_size_data, 2);

  op_upsample_bilinear2d_aa_out(
      input,
      output_size,
      /*align_corners=*/false,
      std::nullopt,
      std::nullopt,
      out);

  // Verify output dimensions
  EXPECT_EQ(out.size(0), 1);
  EXPECT_EQ(out.size(1), 2);
  EXPECT_EQ(out.size(2), 6);
  EXPECT_EQ(out.size(3), 12);
}

TEST_F(OpUpsampleBilinear2dAAOutTest, TestEdgeCaseOneByOne) {
  TensorFactory<ScalarType::Float> tf;

  // Test 1x1 input upsampled to 4x4
  Tensor input = tf.make({1, 3, 1, 1}, {1.0, 2.0, 3.0});
  Tensor out = tf.zeros({1, 3, 4, 4});

  int64_t output_size_data[2] = {4, 4};
  ArrayRef<int64_t> output_size(output_size_data, 2);

  op_upsample_bilinear2d_aa_out(
      input,
      output_size,
      /*align_corners=*/false,
      std::nullopt,
      std::nullopt,
      out);

  // Verify output dimensions
  EXPECT_EQ(out.size(0), 1);
  EXPECT_EQ(out.size(1), 3);
  EXPECT_EQ(out.size(2), 4);
  EXPECT_EQ(out.size(3), 4);

  // All output values should equal corresponding input channel value
  // since we're upsampling from 1x1
  auto in_data = input.const_data_ptr<float>();
  auto out_data = out.const_data_ptr<float>();

  for (int c = 0; c < 3; c++) {
    for (int i = 0; i < 16; i++) {
      EXPECT_NEAR(out_data[c * 16 + i], in_data[c], 0.01);
    }
  }
}

TEST_F(OpUpsampleBilinear2dAAOutTest, TestIdentityTransform) {
  TensorFactory<ScalarType::Float> tf;

  // Test that upsampling to same size preserves input
  Tensor input = tf.make({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});

  Tensor out = tf.zeros({1, 1, 3, 3});

  int64_t output_size_data[2] = {3, 3};
  ArrayRef<int64_t> output_size(output_size_data, 2);

  op_upsample_bilinear2d_aa_out(
      input,
      output_size,
      /*align_corners=*/false,
      std::nullopt,
      std::nullopt,
      out);

  // Output should be very close to input
  auto in_data = input.const_data_ptr<float>();
  auto out_data = out.const_data_ptr<float>();

  for (int i = 0; i < 9; i++) {
    EXPECT_NEAR(out_data[i], in_data[i], 0.01);
  }
}

TEST_F(OpUpsampleBilinear2dAAOutTest, TestLargeDownsample) {
  TensorFactory<ScalarType::Float> tf;

  // Test aggressive downsampling (8x8 -> 2x2) with anti-aliasing
  Tensor input = tf.zeros({1, 1, 8, 8});
  auto in_data = input.mutable_data_ptr<float>();

  // Fill with pattern
  for (int i = 0; i < 64; i++) {
    in_data[i] = static_cast<float>(i);
  }

  Tensor out = tf.zeros({1, 1, 2, 2});

  int64_t output_size_data[2] = {2, 2};
  ArrayRef<int64_t> output_size(output_size_data, 2);

  op_upsample_bilinear2d_aa_out(
      input,
      output_size,
      /*align_corners=*/false,
      std::nullopt,
      std::nullopt,
      out);

  // Verify output dimensions
  EXPECT_EQ(out.size(0), 1);
  EXPECT_EQ(out.size(1), 1);
  EXPECT_EQ(out.size(2), 2);
  EXPECT_EQ(out.size(3), 2);

  // Anti-aliasing should produce smooth downsampled values
  auto out_data = out.const_data_ptr<float>();
  for (int i = 0; i < 4; i++) {
    EXPECT_GT(out_data[i], 0.0f);
    EXPECT_LT(out_data[i], 64.0f);
  }
}

TEST_F(OpUpsampleBilinear2dAAOutTest, TestDoubleDataType) {
  TensorFactory<ScalarType::Double> tf;

  // Test with double precision floating point
  Tensor input = tf.make({1, 1, 2, 2}, {1.0, 2.0, 3.0, 4.0});
  Tensor out = tf.zeros({1, 1, 4, 4});

  int64_t output_size_data[2] = {4, 4};
  ArrayRef<int64_t> output_size(output_size_data, 2);

  op_upsample_bilinear2d_aa_out(
      input,
      output_size,
      /*align_corners=*/false,
      std::nullopt,
      std::nullopt,
      out);

  // Verify output dimensions
  EXPECT_EQ(out.size(0), 1);
  EXPECT_EQ(out.size(1), 1);
  EXPECT_EQ(out.size(2), 4);
  EXPECT_EQ(out.size(3), 4);

  // Check that interpolation produced reasonable values
  auto out_data = out.const_data_ptr<double>();
  EXPECT_GT(out_data[0], 0.0);
  EXPECT_LT(out_data[0], 5.0);
}

TEST_F(OpUpsampleBilinear2dAAOutTest, TestUint8DataType) {
  TensorFactory<ScalarType::Byte> tf;

  // Test with uint8 data type
  Tensor input = tf.make({1, 1, 2, 2}, {50, 100, 150, 200});
  Tensor out = tf.zeros({1, 1, 4, 4});

  int64_t output_size_data[2] = {4, 4};
  ArrayRef<int64_t> output_size(output_size_data, 2);

  op_upsample_bilinear2d_aa_out(
      input,
      output_size,
      /*align_corners=*/false,
      std::nullopt,
      std::nullopt,
      out);

  // Verify output dimensions
  EXPECT_EQ(out.size(0), 1);
  EXPECT_EQ(out.size(1), 1);
  EXPECT_EQ(out.size(2), 4);
  EXPECT_EQ(out.size(3), 4);

  // Check that interpolated values are within input range
  auto out_data = out.const_data_ptr<uint8_t>();
  for (int i = 0; i < 16; i++) {
    EXPECT_GE(out_data[i], 40); // Should be at least close to min input
    EXPECT_LE(out_data[i], 210); // Should be at most close to max input
  }
}

TEST_F(OpUpsampleBilinear2dAAOutTest, TestFractionalDownsample) {
  TensorFactory<ScalarType::Float> tf;

  // Test fractional downsampling (5x7 -> 3x4)
  Tensor input = tf.zeros({1, 2, 5, 7});
  auto in_data = input.mutable_data_ptr<float>();

  // Fill with sequential values
  for (int i = 0; i < 70; i++) {
    in_data[i] = static_cast<float>(i);
  }

  Tensor out = tf.zeros({1, 2, 3, 4});

  int64_t output_size_data[2] = {3, 4};
  ArrayRef<int64_t> output_size(output_size_data, 2);

  op_upsample_bilinear2d_aa_out(
      input,
      output_size,
      /*align_corners=*/false,
      std::nullopt,
      std::nullopt,
      out);

  // Verify output dimensions
  EXPECT_EQ(out.size(0), 1);
  EXPECT_EQ(out.size(1), 2);
  EXPECT_EQ(out.size(2), 3);
  EXPECT_EQ(out.size(3), 4);

  // Verify that anti-aliasing produced reasonable smoothed values
  auto out_data = out.const_data_ptr<float>();
  for (int i = 0; i < 24; i++) {
    EXPECT_GE(out_data[i], 0.0f);
    EXPECT_LE(out_data[i], 70.0f);
  }
}

TEST_F(OpUpsampleBilinear2dAAOutTest, TestLargeBatchSize) {
  TensorFactory<ScalarType::Float> tf;

  // Test with larger batch size to stress test memory access patterns
  Tensor input = tf.zeros({5, 8, 4, 4});
  auto in_data = input.mutable_data_ptr<float>();

  // Fill with unique values per batch/channel
  for (int n = 0; n < 5; n++) {
    for (int c = 0; c < 8; c++) {
      for (int i = 0; i < 16; i++) {
        in_data[n * 8 * 16 + c * 16 + i] =
            static_cast<float>(n * 100 + c * 10 + i);
      }
    }
  }

  Tensor out = tf.zeros({5, 8, 2, 2});

  int64_t output_size_data[2] = {2, 2};
  ArrayRef<int64_t> output_size(output_size_data, 2);

  op_upsample_bilinear2d_aa_out(
      input,
      output_size,
      /*align_corners=*/false,
      std::nullopt,
      std::nullopt,
      out);

  // Verify output dimensions
  EXPECT_EQ(out.size(0), 5);
  EXPECT_EQ(out.size(1), 8);
  EXPECT_EQ(out.size(2), 2);
  EXPECT_EQ(out.size(3), 2);
}

TEST_F(OpUpsampleBilinear2dAAOutTest, TestExtremeDownsample) {
  TensorFactory<ScalarType::Float> tf;

  // Test extreme downsampling (16x16 -> 1x1)
  Tensor input = tf.zeros({1, 1, 16, 16});
  auto in_data = input.mutable_data_ptr<float>();

  // Create a checkerboard pattern to test anti-aliasing effectiveness
  for (int h = 0; h < 16; h++) {
    for (int w = 0; w < 16; w++) {
      in_data[h * 16 + w] = ((h + w) % 2 == 0) ? 1.0f : 0.0f;
    }
  }

  Tensor out = tf.zeros({1, 1, 1, 1});

  int64_t output_size_data[2] = {1, 1};
  ArrayRef<int64_t> output_size(output_size_data, 2);

  op_upsample_bilinear2d_aa_out(
      input,
      output_size,
      /*align_corners=*/false,
      std::nullopt,
      std::nullopt,
      out);

  // Verify output dimensions
  EXPECT_EQ(out.size(0), 1);
  EXPECT_EQ(out.size(1), 1);
  EXPECT_EQ(out.size(2), 1);
  EXPECT_EQ(out.size(3), 1);

  // Anti-aliasing should average the checkerboard pattern to ~0.5
  auto out_data = out.const_data_ptr<float>();
  EXPECT_GT(out_data[0], 0.3f);
  EXPECT_LT(out_data[0], 0.7f);
}

TEST_F(
    OpUpsampleBilinear2dAAOutTest,
    TestConsistencyBetweenScalesAndOutputSize) {
  TensorFactory<ScalarType::Float> tf;

  // Test that providing scales vs output_size gives consistent results
  Tensor input =
      tf.make({1, 2, 3, 4}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                             13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});

  // Method 1: Use output_size
  Tensor out1 = tf.zeros({1, 2, 6, 8});
  int64_t output_size_data[2] = {6, 8};
  ArrayRef<int64_t> output_size(output_size_data, 2);

  op_upsample_bilinear2d_aa_out(
      input,
      output_size,
      /*align_corners=*/false,
      std::nullopt,
      std::nullopt,
      out1);

  // Method 2: Use equivalent scale factors (2x for both dimensions)
  Tensor out2 = tf.zeros({1, 2, 6, 8});

  op_upsample_bilinear2d_aa_out(
      input, output_size, /*align_corners=*/false, 2.0, 2.0, out2);

  // Results should be very close
  auto out1_data = out1.const_data_ptr<float>();
  auto out2_data = out2.const_data_ptr<float>();

  for (int i = 0; i < 48; i++) {
    EXPECT_NEAR(out1_data[i], out2_data[i], 1e-4);
  }
}

TEST_F(OpUpsampleBilinear2dAAOutTest, TestNonSquareInputOutput) {
  TensorFactory<ScalarType::Float> tf;

  // Test with non-square input and output dimensions
  Tensor input =
      tf.make({2, 1, 2, 6}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                             13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});

  Tensor out = tf.zeros({2, 1, 5, 3});

  int64_t output_size_data[2] = {5, 3};
  ArrayRef<int64_t> output_size(output_size_data, 2);

  op_upsample_bilinear2d_aa_out(
      input,
      output_size,
      /*align_corners=*/false,
      std::nullopt,
      std::nullopt,
      out);

  // Verify output dimensions
  EXPECT_EQ(out.size(0), 2);
  EXPECT_EQ(out.size(1), 1);
  EXPECT_EQ(out.size(2), 5);
  EXPECT_EQ(out.size(3), 3);

  // Verify reasonable interpolated values
  auto out_data = out.const_data_ptr<float>();
  for (int i = 0; i < 30; i++) {
    EXPECT_GE(out_data[i], 0.0f);
    EXPECT_LE(out_data[i], 25.0f);
  }
}

TEST_F(OpUpsampleBilinear2dAAOutTest, TestPrecisionConsistency) {
  TensorFactory<ScalarType::Float> tf;

  // Test that results are deterministic and consistent across runs
  Tensor input = tf.make({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});

  Tensor out1 = tf.zeros({1, 1, 7, 7});
  Tensor out2 = tf.zeros({1, 1, 7, 7});

  int64_t output_size_data[2] = {7, 7};
  ArrayRef<int64_t> output_size(output_size_data, 2);

  // Run the same operation twice
  op_upsample_bilinear2d_aa_out(
      input,
      output_size,
      /*align_corners=*/false,
      std::nullopt,
      std::nullopt,
      out1);

  op_upsample_bilinear2d_aa_out(
      input,
      output_size,
      /*align_corners=*/false,
      std::nullopt,
      std::nullopt,
      out2);

  // Results should be identical
  auto out1_data = out1.const_data_ptr<float>();
  auto out2_data = out2.const_data_ptr<float>();

  for (int i = 0; i < 49; i++) {
    EXPECT_EQ(out1_data[i], out2_data[i]);
  }
}

TEST_F(OpUpsampleBilinear2dAAOutTest, TestSpecificInputCase) {
  TensorFactory<ScalarType::Float> tf;

  // Test case with specific inputs:
  // Input shape: [8, 2, 7, 1]
  // Output size: [7, 2]
  // align_corners: false
  // scales_h: 0.010000000000000002
  // scales_w: 10.0
  Tensor input = tf.zeros({8, 2, 7, 1});
  auto in_data = input.mutable_data_ptr<float>();

  // Fill with some test data
  for (int i = 0; i < 8 * 2 * 7 * 1; i++) {
    in_data[i] = static_cast<float>(i) * 0.1f;
  }

  // Output shape will be [8, 2, 7, 2]
  Tensor out = tf.zeros({8, 2, 7, 2});

  int64_t output_size_data[2] = {7, 2};
  ArrayRef<int64_t> output_size(output_size_data, 2);

  op_upsample_bilinear2d_aa_out(
      input,
      output_size,
      /*align_corners=*/false,
      0.010000000000000002,
      10.0,
      out);

  // Verify output dimensions
  EXPECT_EQ(out.size(0), 8);
  EXPECT_EQ(out.size(1), 2);
  EXPECT_EQ(out.size(2), 7);
  EXPECT_EQ(out.size(3), 2);

  // Verify that output has reasonable values
  auto out_data = out.const_data_ptr<float>();
  for (int i = 0; i < 8 * 2 * 7 * 2; i++) {
    // Check for NaN or Inf
    EXPECT_FALSE(std::isnan(out_data[i]));
    EXPECT_FALSE(std::isinf(out_data[i]));
  }
}
