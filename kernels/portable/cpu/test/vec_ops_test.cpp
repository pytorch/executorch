/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/vec_ops.h>

#include <limits>
#include <vector>

#include <gtest/gtest.h>

using namespace ::testing;

TEST(VecMinfTest, Smoke) {
  // No need to be super thorough since we know this is implemented with
  // std::min_element(). Just show that it's hooked up correctly.

  constexpr size_t kNumVals = 5;
  float x[kNumVals] = {1.1, -2.2, 0, -1234.5, 10.0};
  EXPECT_EQ(torch::executor::vec_minf(x, kNumVals), -1234.5);
}

TEST(VecMaxfTest, Smoke) {
  // No need to be super thorough since we know this is implemented with
  // std::max_element(). Just show that it's hooked up correctly.

  constexpr size_t kNumVals = 5;
  float x[kNumVals] = {1.1, -2.2, 0, -1234.5, 10.0};
  EXPECT_EQ(torch::executor::vec_maxf(x, kNumVals), 10.0);
}

TEST(VecAddfTest, Smoke) {
  constexpr size_t kNumVals = 5;
  float in1[kNumVals] = {1, 2, 3, 4, 5};
  float in2[kNumVals] = {10, 20, 30, 40, 50};
  float out[kNumVals] = {};

  torch::executor::vec_addf(out, in1, in2, kNumVals);

  // Each element of `out` should be the sum of the corresponding elements
  // of `in1` and `in2`.
  EXPECT_EQ(out[0], 11);
  EXPECT_EQ(out[1], 22);
  EXPECT_EQ(out[2], 33);
  EXPECT_EQ(out[3], 44);
  EXPECT_EQ(out[4], 55);
}

TEST(VecScalefTest, Smoke) {
  constexpr size_t kNumVals = 5;
  float in[kNumVals] = {4, 8, 16, 32, 64};
  float out[kNumVals] = {0, 0, 0, 0, 0};

  torch::executor::vec_scalef(out, in, 0.5, kNumVals);

  // Each element of `out` should be the product of 0.5 and the corresponding
  // element of `in`.
  EXPECT_EQ(out[0], 2);
  EXPECT_EQ(out[1], 4);
  EXPECT_EQ(out[2], 8);
  EXPECT_EQ(out[3], 16);
  EXPECT_EQ(out[4], 32);
}

TEST(VecPowerfTest, Smoke) {
  constexpr size_t kNumVals = 5;
  float in[kNumVals] = {-2, -1, 0, 1, 2};

  // Should return the sum of the squares of all input elements.
  EXPECT_EQ(
      torch::executor::vec_powerf(in, kNumVals),
      (-2 * -2) + (-1 * -1) + (0 * 0) + (1 * 1) + (2 * 2));
}

TEST(VecMatMulTest, Smoke) {
  // x sizes: (3, 2)
  constexpr size_t kXNumVals = 6;
  // y sizes: (2, 4)
  constexpr size_t kYNumVals = 8;
  // z sizes: (3, 4)
  constexpr size_t kZNumVals = 12;

  // clang-format off
  int64_t X[kXNumVals] = {
    1, 2,
    2, 1,
    3, 0,
  };
  int64_t Y[kYNumVals] = {
    1, 2, 3, 4,
    5, 6, 7, 8,
  };
  // clang-format on

  int64_t out[kZNumVals] = {};

  torch::executor::vec_matmul(out, X, Y, 3, 2, 4);

  // clang-format off
  std::vector<int64_t> expected({
    11, 14, 17, 20,
    7, 10, 13, 16,
    3, 6, 9, 12,
  });
  // clang-format on
  EXPECT_EQ(std::vector<int64_t>(out, out + kZNumVals), expected);
}

TEST(VecAddmmTest, Smoke) {
  // x sizes: (3, 2)
  constexpr size_t kXNumVals = 6;
  // y sizes: (2, 4)
  constexpr size_t kYNumVals = 8;
  // z sizes: (3, 4)
  constexpr size_t kZNumVals = 12;

  // clang-format off
  int64_t self[kZNumVals] = {
    1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
  };

  int64_t X[kXNumVals] = {
    1, 2,
    2, 1,
    3, 0,
  };
  int64_t Y[kYNumVals] = {
    1, 2, 3, 4,
    5, 6, 7, 8,
  };
  // clang-format on

  int64_t out[kZNumVals] = {};

  torch::executor::vec_addmm(out, self, X, Y, 3, 2, 4, 2.0, 3.0);

  // clang-format off
  std::vector<int64_t> expected({
    35, 44, 53, 62,
    25, 34, 43, 52,
    15, 24, 33, 42,
  });
  // clang-format on
  EXPECT_EQ(std::vector<int64_t>(out, out + kZNumVals), expected);
}

TEST(VecSoftMaxTest, Smoke) {
  // x sizes: (1, 3)
  constexpr size_t kXNumVals = 3;

  // clang-format off
  float X[kXNumVals] = {
    1, 2, 3,
  };
  // clang-format on
  float out[kXNumVals] = {};

  torch::executor::vec_softmax(out, X, 3);

  // clang-format off
  std::vector<float> expected({
      0.0900306, 0.244728, 0.665241,
  });
  // clang-format on
  for (auto i = 0; i < 3; ++i) {
    EXPECT_NEAR(out[i], expected[i], 10e-6);
  }
}

class QuantizeI8F32Test : public ::testing::Test {
 protected:
  void SetUp() override {
    constexpr float kInfinity = std::numeric_limits<double>::infinity();
    // A spread of inputs for various scales and zero points.
    inputs_ = {
        -kInfinity, -512, -256, -128, -64, 0, 64, 128, 256, 512, kInfinity};
    outputs_.resize(inputs_.size());
  }

  std::vector<float> inputs_;
  std::vector<int8_t> outputs_;
};

TEST_F(QuantizeI8F32Test, Identity) {
  const float kScale = 1.0; // No scaling.
  const int32_t kZeroPoint = 0; // Not shifted.
  torch::executor::quantize_i8_f32(
      outputs_.data(), inputs_.data(), kScale, kZeroPoint, inputs_.size());

  // Most values will be clamped to min/max uint8_t, but the unclamped values
  // should be the same as the inputs.
  EXPECT_EQ(
      outputs_,
      std::vector<int8_t>(
          {-128, -128, -128, -128, -64, 0, 64, 127, 127, 127, 127}));
}

TEST_F(QuantizeI8F32Test, Rounding) {
  // Demonstrate that quantization uses roundf() semantics, not
  // ceilf()/floorf().
  std::vector<float> in = {-1.9, -1.1, 1.1, 1.9};
  std::vector<int8_t> out;
  out.resize(in.size());

  torch::executor::quantize_i8_f32(
      out.data(), in.data(), /*scale=*/1.0, /*zero_point=*/0, in.size());

  EXPECT_EQ(out, std::vector<int8_t>({-2, -1, 1, 2}));
}

TEST_F(QuantizeI8F32Test, ScaledDown) {
  const float kScale = 0.5; // Scaled down.
  const int32_t kZeroPoint = 0; // Not shifted.
  torch::executor::quantize_i8_f32(
      outputs_.data(), inputs_.data(), kScale, kZeroPoint, inputs_.size());

  EXPECT_EQ(
      outputs_,
      std::vector<int8_t>({
          -128, // Clamped
          -128, // Clamped
          -128, // Clamped
          -64, // -128 * 0.5
          -32, // -64 * 0.5
          0, // 0 * 0.5
          32, // 64 * 0.5
          64, // 128 * 0.5
          127, // Clamped
          127, // Clamped
          127, // Clamped
      }));
}

TEST_F(QuantizeI8F32Test, ShiftedZeroPoint) {
  const float kScale = 1.0; // No scaling.
  const int32_t kZeroPoint = 32; // Shifted.
  torch::executor::quantize_i8_f32(
      outputs_.data(), inputs_.data(), kScale, kZeroPoint, inputs_.size());

  EXPECT_EQ(
      outputs_,
      std::vector<int8_t>({
          -128, // Clamped
          -128, // Clamped
          -128, // Clamped
          -96, // -128 + 32
          -32, // -64 + 32
          32, // 0 + 32
          96, // 64 + 32
          127, // Clamped
          127, // Clamped
          127, // Clamped
          127, // Clamped
      }));
}

TEST_F(QuantizeI8F32Test, ScaledDownWithShiftedZeroPoint) {
  const float kScale = 0.5; // Scaled down.
  const int32_t kZeroPoint = 32; // Shifted.
  torch::executor::quantize_i8_f32(
      outputs_.data(), inputs_.data(), kScale, kZeroPoint, inputs_.size());

  // Demonstrate that the zero point adjustment happens after scaling.
  EXPECT_EQ(
      outputs_,
      std::vector<int8_t>({
          -128, // Clamped
          -128, // Clamped
          -96, // (-256 * 0.5) + 32
          -32, // (-128 * 0.5) + 32
          0, // (-64 * 0.5) + 32
          32, // (0 * 0.5) + 32
          64, // (64 * 0.5) + 32
          96, // (128 * 0.5) + 32
          127, // Clamped
          127, // Clamped
          127, // Clamped
      }));
}

class DequantizeI8F32Test : public ::testing::Test {
 protected:
  void SetUp() override {
    // A spread of inputs for various scales and zero points.
    inputs_ = {-128, -64, -32, 0, 32, 64, 127};
    outputs_.resize(inputs_.size());
  }

  std::vector<int8_t> inputs_;
  std::vector<float> outputs_;
};

TEST_F(DequantizeI8F32Test, Identity) {
  const float kScale = 1.0; // No scaling.
  const int32_t kZeroPoint = 0; // Not shifted.
  torch::executor::dequantize_i8_f32(
      outputs_.data(), inputs_.data(), kScale, kZeroPoint, inputs_.size());

  EXPECT_EQ(outputs_, std::vector<float>({-128, -64, -32, 0, 32, 64, 127}));
}

TEST_F(DequantizeI8F32Test, ScaledUp) {
  const float kScale = 2.0; // Scaled up.
  const int32_t kZeroPoint = 0; // Not shifted.
  torch::executor::dequantize_i8_f32(
      outputs_.data(), inputs_.data(), kScale, kZeroPoint, inputs_.size());

  EXPECT_EQ(
      outputs_,
      std::vector<float>(
          {-128 * kScale,
           -64 * kScale,
           -32 * kScale,
           0 * kScale,
           32 * kScale,
           64 * kScale,
           127 * kScale}));
}

TEST_F(DequantizeI8F32Test, ShiftedZeroPoint) {
  const float kScale = 1.0; // Not scaled.
  const int32_t kZeroPoint = 32; // Shifted.
  torch::executor::dequantize_i8_f32(
      outputs_.data(), inputs_.data(), kScale, kZeroPoint, inputs_.size());

  EXPECT_EQ(
      outputs_,
      std::vector<float>(
          {-128 - kZeroPoint,
           -64 - kZeroPoint,
           -32 - kZeroPoint,
           0 - kZeroPoint,
           32 - kZeroPoint,
           64 - kZeroPoint,
           127 - kZeroPoint}));
}

TEST_F(DequantizeI8F32Test, ScaledUpWithShiftedZeroPoint) {
  const float kScale = 2.0; // Scaled up.
  const int32_t kZeroPoint = 32; // Shifted.
  torch::executor::dequantize_i8_f32(
      outputs_.data(), inputs_.data(), kScale, kZeroPoint, inputs_.size());

  // Demonstrate that the zero point adjustment happens before scaling.
  EXPECT_EQ(
      outputs_,
      std::vector<float>({
          (-128 - kZeroPoint) * kScale,
          (-64 - kZeroPoint) * kScale,
          (-32 - kZeroPoint) * kScale,
          (0 - kZeroPoint) * kScale,
          (32 - kZeroPoint) * kScale,
          (64 - kZeroPoint) * kScale,
          (127 - kZeroPoint) * kScale,
      }));
}
