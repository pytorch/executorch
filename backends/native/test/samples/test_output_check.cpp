// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/test/samples/OutputCheck.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
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
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn::testing
