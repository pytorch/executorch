// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/test/samples/OutputCheck.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace ptn::testing {
namespace {

constexpr size_t kCount = 4096;
constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

SampleTensor f32(const std::vector<float>& values) {
  SampleTensor t{
      .dtype = kFloat, .sizes = {static_cast<int64_t>(values.size())}};
  t.bytes.resize(values.size() * sizeof(float));
  std::memcpy(t.bytes.data(), values.data(), t.bytes.size());
  return t;
}

SampleTensor bits16(ScalarType dtype, const std::vector<uint16_t>& bits) {
  SampleTensor t{.dtype = dtype, .sizes = {static_cast<int64_t>(bits.size())}};
  for (const uint16_t b : bits) {
    t.bytes.push_back(b & 0xFF);
    t.bytes.push_back(b >> 8);
  }
  return t;
}

std::vector<float> normal(double stddev, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> dist(0.0, stddev);
  std::vector<float> out(kCount);
  for (float& v : out) {
    v = static_cast<float>(dist(rng));
  }
  return out;
}

// x * scale + noise + bias, elementwise.
std::vector<float> affine(
    const std::vector<float>& x,
    double scale,
    const std::vector<float>& noise,
    double bias = 0.0) {
  std::vector<float> out(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    out[i] = static_cast<float>(x[i] * scale + noise[i] + bias);
  }
  return out;
}

std::vector<std::string> failed_rules(
    const std::vector<float>& got,
    const std::vector<float>& golden,
    const std::vector<float>& precise,
    const Policy& policy = {}) {
  OutputCheck check;
  check.add(got, f32(golden), f32(precise));
  std::vector<std::string> rules;
  for (const std::string& failure : check.failures(policy)) {
    rules.push_back(failure.substr(0, failure.find(':')));
  }
  return rules;
}

// fp16's round to nearest even of v, without overflow.
double fp16(double v) {
  int exponent = 0;
  std::frexp(v, &exponent);
  const double ulp = std::ldexp(1.0, std::max(exponent, -13) - 11);
  return std::nearbyint(v / ulp) * ulp;
}

// The fp16 bits of a value fp16 holds exactly.
uint16_t half_bits(double v) {
  const uint16_t sign = std::signbit(v) ? 0x8000 : 0;
  v = std::abs(v);
  if (v < 0x1p-14) {
    return sign | static_cast<uint16_t>(v * 0x1p24);
  }
  int exponent = 0;
  const double fraction = std::frexp(v, &exponent);
  return sign | static_cast<uint16_t>((exponent + 14) << 10) |
      static_cast<uint16_t>((fraction * 2 - 1) * 1024);
}

// Q4_1 weights whose blocks scale d and m by 1/2, 1 or 2, so that a decoder
// reading another block's scales does not match, against activations in
// [-1, 1], with ggml's GPU dot products for up to 8 rows.
class BlockScaleTest : public ::testing::Test {
 protected:
  static constexpr int64_t kOut = 16;
  static constexpr int64_t kIn = 256;
  static constexpr int64_t kRows = 4;
  static constexpr int64_t kBlocks = kIn / 32;

  void SetUp() override {
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> code(0, 15);
    std::uniform_int_distribution<int> factor(-1, 1);
    std::uniform_real_distribution<float> value(-1.0f, 1.0f);
    for (int64_t b = 0; b < kOut * kBlocks; ++b) {
      d_.push_back(std::ldexp(0x1p-13, factor(rng)));
      m_.push_back(std::ldexp(-15 * 0x1p-14, factor(rng)));
      const uint16_t bits[2] = {half_bits(d_.back()), half_bits(m_.back())};
      for (const uint16_t h : bits) {
        weight_.push_back(h & 0xFF);
        weight_.push_back(h >> 8);
      }
      std::vector<int> codes(32);
      for (int& q : codes) {
        q = code(rng);
      }
      codes_.push_back(codes);
      for (int j = 0; j < 16; ++j) {
        weight_.push_back(codes[j] | (codes[j + 16] << 4));
      }
    }
    for (int64_t i = 0; i < kRows * kIn; ++i) {
      x_.push_back(value(rng));
    }
  }

  // The GPU's outputs, its weight scales and mins taken from the block
  // `shift_d` and `shift_m` further along the row; precise ones without
  // activation quantization when `precise`.
  std::vector<float> outputs(int shift_d, int shift_m, bool precise = false)
      const {
    std::vector<float> out(kRows * kOut);
    for (int64_t r = 0; r < kRows; ++r) {
      for (int64_t b = 0; b < kBlocks; ++b) {
        const float* x = &x_[r * kIn + b * 32];
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
          amax = std::max(amax, std::abs(x[i]));
        }
        const float d8 = amax / 127.0f;
        std::vector<int> q8(32);
        int sum = 0;
        for (int i = 0; i < 32; ++i) {
          q8[i] = static_cast<int>(std::round(x[i] / d8));
          sum += q8[i];
        }
        for (int64_t o = 0; o < kOut; ++o) {
          const int64_t block = o * kBlocks + b;
          const int64_t row = o * kBlocks;
          const double d = d_[row + (b + shift_d) % kBlocks];
          const double m = m_[row + (b + shift_m) % kBlocks];
          double sum_qx = 0.0;
          int dot = 0;
          for (int i = 0; i < 32; ++i) {
            sum_qx += codes_[block][i] * static_cast<double>(x[i]);
            dot += codes_[block][i] * q8[i];
          }
          double sum_x = 0.0;
          for (int i = 0; i < 32; ++i) {
            sum_x += x[i];
          }
          out[r * kOut + o] += static_cast<float>(
              precise ? d * sum_qx + m * sum_x
                      : dot * fp16(d * fp16(d8)) +
                      fp16(m * fp16(d8 * static_cast<float>(sum))));
        }
      }
    }
    return out;
  }

  BlockScaleLinear linear(bool vector_kernel = true) const {
    const std::vector<double> x(x_.begin(), x_.end());
    return block_scale_linear(
        "gguf:q4_1", weight_, kOut, kIn, x, {}, vector_kernel);
  }

  std::vector<std::string> failures(
      const std::vector<float>& got,
      std::span<const double> allowance) const {
    const BlockScaleLinear exact = linear();
    OutputCheck check;
    check.add(
        got,
        f32({exact.outputs.begin(), exact.outputs.end()}),
        f32(outputs(0, 0, /*precise=*/true)),
        allowance);
    return check.failures({});
  }

  std::vector<double> d_;
  std::vector<double> m_;
  std::vector<std::vector<int>> codes_;
  std::vector<uint8_t> weight_;
  std::vector<float> x_;
};

class OutputCheckTest : public ::testing::Test {
 protected:
  const std::vector<float> precise_ = normal(1.0, 0);
  const std::vector<float> golden_ = affine(precise_, 1.0, normal(1e-3, 1));
  const std::vector<float> engine_noise_ = normal(1e-3, 2);
};

// cppcheck-suppress-begin syntaxError
TEST(ToFloatTest, DecodesHalfAndBFloat16) {
  const std::vector<float> half = {1.5f, -2.0f, std::ldexp(1.0f, -24), -kInf};
  EXPECT_EQ(to_float(bits16(kHalf, {0x3E00, 0xC000, 0x0001, 0xFC00})), half);
  const std::vector<float> bf16 = {1.5f, -2.0f, kInf};
  EXPECT_EQ(to_float(bits16(kBFloat16, {0x3FC0, 0xC000, 0x7F80})), bf16);
}

TEST(ToFloatTest, NarrowsDouble) {
  const std::vector<double> values = {1.5, -2.0, 0.1};
  SampleTensor t{.dtype = kDouble, .sizes = {3}};
  t.bytes.resize(values.size() * sizeof(double));
  std::memcpy(t.bytes.data(), values.data(), t.bytes.size());
  const std::vector<float> expected = {1.5f, -2.0f, 0.1f};
  EXPECT_EQ(to_float(t), expected);
}

TEST(ToFloatTest, RejectsIntegers) {
  EXPECT_THROW(to_float(bits16(kShort, {1})), std::invalid_argument);
}

TEST_F(OutputCheckTest, PassesNoiseLikeEagers) {
  EXPECT_EQ(
      failed_rules(affine(precise_, 1.0, engine_noise_), golden_, precise_),
      std::vector<std::string>{});
}

TEST_F(OutputCheckTest, PassesUlpErrorsAgainstAnExactEager) {
  std::vector<float> got = precise_;
  for (float& v : got) {
    v = std::nextafter(v, kInf);
  }
  EXPECT_EQ(failed_rules(got, precise_, precise_), std::vector<std::string>{});
}

TEST_F(OutputCheckTest, PassesBiasesSharedWithEager) {
  const std::vector<float> golden =
      affine(precise_, 1.002, normal(1e-3, 1), 1e-3);
  EXPECT_EQ(
      failed_rules(
          affine(precise_, 1.002, engine_noise_, 1e-3), golden, precise_),
      std::vector<std::string>{});
}

TEST_F(OutputCheckTest, FailsNoiseFarAboveEagers) {
  const std::vector<std::string> expected = {"rmse", "max_abs"};
  EXPECT_EQ(
      failed_rules(affine(precise_, 1.0, normal(1e-2, 2)), golden_, precise_),
      expected);
}

TEST_F(OutputCheckTest, PassesNoiseFarAboveEagersUnderLooserRatios) {
  EXPECT_EQ(
      failed_rules(
          affine(precise_, 1.0, normal(1e-2, 2)),
          golden_,
          precise_,
          Policy{.rmse_ratio = 20.0, .max_ratio = 20.0}),
      std::vector<std::string>{});
}

TEST_F(OutputCheckTest, FailsOneOutlier) {
  std::vector<float> precise = precise_;
  std::vector<float> golden = golden_;
  precise[7] = golden[7] = 0.0f;
  std::vector<float> got = affine(precise, 1.0, engine_noise_);
  got[7] = 0.1f;
  EXPECT_EQ(
      failed_rules(got, golden, precise), std::vector<std::string>{"max_abs"});
}

TEST_F(OutputCheckTest, FailsSmallAbsoluteBias) {
  EXPECT_EQ(
      failed_rules(
          affine(precise_, 1.0, engine_noise_, 1e-3), golden_, precise_),
      std::vector<std::string>{"abs_bias"});
}

TEST_F(OutputCheckTest, FailsSmallRelativeBias) {
  EXPECT_EQ(
      failed_rules(affine(precise_, 1.002, engine_noise_), golden_, precise_),
      std::vector<std::string>{"rel_bias"});
}

TEST_F(OutputCheckTest, ExcludesNonFiniteMatchingGolden) {
  std::vector<float> precise = precise_;
  std::vector<float> golden = golden_;
  std::vector<float> got = affine(precise_, 1.0, engine_noise_);
  golden[0] = got[0] = kNaN;
  golden[1] = got[1] = -kInf;
  precise[2] = kInf;
  got[2] = golden[2];

  OutputCheck check;
  check.add(got, f32(golden), f32(precise));

  EXPECT_EQ(check.failures(Policy{}), std::vector<std::string>{});
  EXPECT_EQ(check.engine().count, kCount - 3);
}

TEST_F(OutputCheckTest, FailsNonFiniteDifferingFromGolden) {
  std::vector<float> precise = precise_;
  std::vector<float> golden = golden_;
  std::vector<float> got = affine(precise_, 1.0, engine_noise_);
  golden[0] = kInf;
  got[0] = -kInf;
  got[1] = kNaN;
  golden[2] = kNaN;
  precise[3] = kNaN;

  OutputCheck check;
  check.add(got, f32(golden), f32(precise));

  const std::vector<std::string> expected = {
      "nonfinite: 4 values differ from golden where a value is NaN or infinite"};
  EXPECT_EQ(check.failures(Policy{}), expected);
}

TEST_F(OutputCheckTest, AccumulatesAcrossSamples) {
  const std::vector<float> got = affine(precise_, 1.0, engine_noise_, 1e-3);
  const size_t half = kCount / 2;
  const auto part = [&](const std::vector<float>& v, size_t begin) {
    return std::vector<float>(v.begin() + begin, v.begin() + begin + half);
  };

  OutputCheck whole;
  whole.add(got, f32(golden_), f32(precise_));
  OutputCheck split;
  for (const size_t begin : {size_t{0}, half}) {
    split.add(
        part(got, begin),
        f32(part(golden_, begin)),
        f32(part(precise_, begin)));
  }

  EXPECT_EQ(split.engine().count, whole.engine().count);
  EXPECT_DOUBLE_EQ(split.engine().abs_bias, whole.engine().abs_bias);
  EXPECT_DOUBLE_EQ(split.engine().rel_bias_se, whole.engine().rel_bias_se);
}

TEST(OutputCheckFitTest, MatchesClosedFormOnAnOffsetPrecise) {
  // d = got - precise = 1 + 0.5 * precise + e, with e orthogonal to 1 and to
  // precise, so the fit is exact and the residual variance is |e|^2 / (n - 2).
  const std::vector<float> precise = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  const std::vector<float> got = {2.5f, 4.25f, 5.0f, 7.25f, 8.5f};
  OutputCheck check;
  check.add(got, f32(precise), f32(precise));

  const ErrorStats stats = check.engine();
  const double residual = 0.375 / 3;
  EXPECT_EQ(stats.count, 5);
  EXPECT_NEAR(stats.rmse, std::sqrt(34.125 / 5), 1e-12);
  EXPECT_EQ(stats.max_abs, 3.5);
  EXPECT_NEAR(stats.abs_bias, 1.0, 1e-12);
  EXPECT_NEAR(stats.rel_bias, 0.5, 1e-12);
  EXPECT_NEAR(
      stats.abs_bias_se, std::sqrt(residual * (1.0 / 5 + 9.0 / 10)), 1e-12);
  EXPECT_NEAR(stats.rel_bias_se, std::sqrt(residual / 10), 1e-12);
}

TEST(OutputCheckSizeTest, RejectsMismatchedCounts) {
  OutputCheck check;
  EXPECT_THROW(
      check.add(std::vector<float>{1.0f}, f32({1.0f, 2.0f}), f32({1.0f})),
      std::invalid_argument);
}
TEST_F(BlockScaleTest, BoundsTheGpuProducts) {
  const BlockScaleLinear exact = linear();
  const std::vector<float> gpu = outputs(0, 0);
  double worst = 0.0;
  for (size_t i = 0; i < gpu.size(); ++i) {
    // Slack for float accumulation.
    EXPECT_LE(std::abs(gpu[i] - exact.outputs[i]), exact.bounds[i] + 1e-6) << i;
    worst = std::max(worst, std::abs(gpu[i] - exact.outputs[i]));
  }
  EXPECT_GT(worst, 1e-5);
}

TEST_F(BlockScaleTest, AllowsTheGpuProducts) {
  const std::vector<float> gpu = outputs(0, 0);
  EXPECT_FALSE(failures(gpu, {}).empty());
  EXPECT_EQ(failures(gpu, linear().bounds), std::vector<std::string>());
}

TEST_F(BlockScaleTest, CatchesAnotherBlocksScaleOrMin) {
  const std::vector<double> bounds = linear().bounds;
  EXPECT_FALSE(failures(outputs(1, 0), bounds).empty());
  EXPECT_FALSE(failures(outputs(0, 1), bounds).empty());
}

TEST_F(BlockScaleTest, AllowsNothingForOtherKernels) {
  for (const double bound : linear(/*vector_kernel=*/false).bounds) {
    EXPECT_EQ(bound, 0.0);
  }
}

TEST_F(BlockScaleTest, AddsTheErrorOfItsInputs) {
  std::vector<double> x(x_.begin(), x_.end());
  std::vector<double> x_error(x.size(), 1e-3);
  const BlockScaleLinear chained =
      block_scale_linear("gguf:q4_1", weight_, kOut, kIn, x, x_error, true);
  const BlockScaleLinear exact = linear();
  for (int64_t o = 0; o < kOut; ++o) {
    double row_norm = 0.0;
    for (int64_t b = 0; b < kBlocks; ++b) {
      for (const int q : codes_[o * kBlocks + b]) {
        row_norm += std::abs(d_[o * kBlocks + b] * q + m_[o * kBlocks + b]);
      }
    }
    EXPECT_GE(chained.bounds[o] - exact.bounds[o], 1e-3 * row_norm * 0.999);
  }
}

TEST(BlockScaleLinearTest, RejectsOtherCodecsAndSizes) {
  const std::vector<uint8_t> weight(20);
  const std::vector<double> x(32);
  EXPECT_THROW(
      block_scale_linear("gguf:q4_0", weight, 1, 32, x, {}, true),
      std::invalid_argument);
  EXPECT_THROW(
      block_scale_linear("gguf:q4_1", weight, 1, 64, x, {}, true),
      std::invalid_argument);
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn::testing
