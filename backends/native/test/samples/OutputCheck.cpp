// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/test/samples/OutputCheck.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace ptn::testing {
namespace {

uint32_t load_u16(const uint8_t* bytes) {
  return static_cast<uint32_t>(bytes[0]) |
      (static_cast<uint32_t>(bytes[1]) << 8);
}

float half_to_float(const uint8_t* bytes) {
  const uint32_t h = load_u16(bytes);
  const uint32_t sign = (h & 0x8000u) << 16;
  const uint32_t exponent = (h >> 10) & 0x1Fu;
  const uint32_t mantissa = h & 0x3FFu;
  if (exponent == 0) {
    const float magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    return sign != 0 ? -magnitude : magnitude;
  }
  if (exponent == 0x1Fu) {
    return std::bit_cast<float>(sign | 0x7F800000u | (mantissa << 13));
  }
  return std::bit_cast<float>(
      sign | ((exponent + 112) << 23) | (mantissa << 13));
}

bool same(double a, double b) {
  return a == b || (std::isnan(a) && std::isnan(b));
}

// Half the fp16 ulp at v, for |v| up to fp16's largest finite value.
double half_fp16_ulp(double v) {
  if (std::abs(v) < 0x1p-14) {
    return 0x1p-25;
  }
  int exponent = 0;
  std::frexp(v, &exponent);
  return std::ldexp(1.0, exponent - 12);
}

double round_to_fp16(double v) {
  return std::nearbyint(v / (2 * half_fp16_ulp(v))) * 2 * half_fp16_ulp(v);
}

constexpr int64_t kBlock = 32;

// The 32 codes, d and m of one Q4_1 or Q5_1 block.
struct WeightBlock {
  double d = 0.0;
  double m = 0.0;
  int codes[kBlock] = {};
};

WeightBlock weight_block(const uint8_t* block, bool fifth_bit) {
  WeightBlock out{.d = half_to_float(block), .m = half_to_float(block + 2)};
  const uint8_t* nibbles = block + (fifth_bit ? 8 : 4);
  uint32_t high = 0;
  if (fifth_bit) {
    std::memcpy(&high, block + 4, sizeof(high));
  }
  for (int j = 0; j < 16; ++j) {
    out.codes[j] = (nibbles[j] & 0xF) | (((high >> j) & 1) << 4);
    out.codes[j + 16] = (nibbles[j] >> 4) | (((high >> (j + 16)) & 1) << 4);
  }
  return out;
}

// One activation block as ggml's GPU Q8_1 quantizer stores it, with `step`
// the most each code may differ by and the largest |d8| and |s| that allows.
struct ActivationBlock {
  double d8 = 0.0;
  double s = 0.0;
  int q8[kBlock] = {};
  double step = 0.0;
  double d8_max = 0.0;
  double s_max = 0.0;
};

ActivationBlock activation_block(const double* x, double x_error) {
  float amax = 0.0f;
  for (int64_t i = 0; i < kBlock; ++i) {
    amax = std::max(amax, std::abs(static_cast<float>(x[i])));
  }
  const float d8 = amax / 127.0f;
  ActivationBlock out;
  int sum = 0;
  for (int64_t i = 0; i < kBlock; ++i) {
    out.q8[i] = amax == 0.0f
        ? 0
        : static_cast<int>(std::round(static_cast<float>(x[i]) / d8));
    sum += out.q8[i];
  }
  out.d8 = round_to_fp16(d8);
  out.s = round_to_fp16(static_cast<double>(d8 * static_cast<float>(sum)));
  out.d8_max = round_to_fp16((amax + x_error) / 127.0 * (1 + 0x1p-10));
  out.step = d8 > 0.0f ? x_error / d8 + 1 : 1;
  out.s_max = out.d8_max * (std::abs(sum) + kBlock * out.step);
  return out;
}

} // namespace

BlockScaleLinear block_scale_linear(
    std::string_view codec,
    std::span<const uint8_t> weight,
    int64_t out,
    int64_t in,
    std::span<const double> x,
    std::span<const double> x_error,
    bool vector_kernel) {
  if (codec != "gguf:q4_1" && codec != "gguf:q5_1") {
    throw std::invalid_argument(
        "block_scale_linear: unsupported codec " + std::string(codec));
  }
  const bool fifth_bit = codec == "gguf:q5_1";
  const int64_t block_bytes = fifth_bit ? 24 : 20;
  const int64_t blocks = in / kBlock;
  if (out <= 0 || in <= 0 || in % kBlock != 0 ||
      std::cmp_not_equal(weight.size(), out * blocks * block_bytes) ||
      x.size() % in != 0 || (!x_error.empty() && x_error.size() != x.size())) {
    throw std::invalid_argument("block_scale_linear: mismatched sizes");
  }
  const int64_t rows = static_cast<int64_t>(x.size()) / in;
  std::vector<ActivationBlock> acts(rows * blocks);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t b = 0; b < blocks; ++b) {
      double error = 0.0;
      for (int64_t i = 0; !x_error.empty() && i < kBlock; ++i) {
        error = std::max(error, x_error[r * in + b * kBlock + i]);
      }
      acts[r * blocks + b] = activation_block(&x[r * in + b * kBlock], error);
    }
  }
  BlockScaleLinear result{
      .outputs = std::vector<double>(rows * out),
      .bounds = std::vector<double>(rows * out)};
  for (int64_t o = 0; o < out; ++o) {
    for (int64_t b = 0; b < blocks; ++b) {
      const WeightBlock w =
          weight_block(&weight[(o * blocks + b) * block_bytes], fifth_bit);
      const int code_sum =
          std::accumulate(std::begin(w.codes), std::end(w.codes), 0);
      for (int64_t r = 0; r < rows; ++r) {
        const ActivationBlock& a = acts[r * blocks + b];
        int dot = 0;
        double propagated = 0.0;
        for (int64_t i = 0; i < kBlock; ++i) {
          dot += w.codes[i] * a.q8[i];
          if (!x_error.empty()) {
            propagated += std::abs(w.d * w.codes[i] + w.m) *
                x_error[r * in + b * kBlock + i];
          }
        }
        result.outputs[r * out + o] += dot * w.d * a.d8 + w.m * a.s;
        double& bound = result.bounds[r * out + o];
        bound += propagated;
        if (vector_kernel) {
          bound += (std::abs(dot) + code_sum * a.step) *
                  half_fp16_ulp(std::abs(w.d) * a.d8_max) +
              half_fp16_ulp(std::abs(w.m) * a.s_max);
        }
      }
    }
  }
  return result;
}

namespace {

std::string g3(double v) {
  char text[32];
  std::snprintf(text, sizeof(text), "%.3g", v);
  return text;
}

} // namespace

std::vector<float> to_float(const SampleTensor& t) {
  const size_t numel = t.bytes.size() / element_size(t.dtype);
  std::vector<float> out(numel);
  const uint8_t* bytes = t.bytes.data();
  switch (t.dtype) {
    case kFloat:
      std::memcpy(out.data(), bytes, numel * sizeof(float));
      break;
    case kDouble:
      for (size_t i = 0; i < numel; ++i) {
        double v;
        std::memcpy(&v, bytes + i * sizeof(double), sizeof(double));
        out[i] = static_cast<float>(v);
      }
      break;
    case kHalf:
      for (size_t i = 0; i < numel; ++i) {
        out[i] = half_to_float(bytes + 2 * i);
      }
      break;
    case kBFloat16:
      for (size_t i = 0; i < numel; ++i) {
        out[i] = std::bit_cast<float>(load_u16(bytes + 2 * i) << 16);
      }
      break;
    case kByte:
    case kChar:
    case kShort:
    case kInt:
    case kLong:
    case kBool:
    case kUInt16:
    case kUInt32:
    case kUInt64:
      throw std::invalid_argument(
          std::string("to_float: unsupported dtype ") +
          scalar_type_name(t.dtype));
  }
  return out;
}

// Welford co-moments, stable when the means are large against the spread.
void OutputCheck::Fit::add(double p, double d) {
  ++n;
  const double dp = p - mean_p;
  const double dd = d - mean_d;
  mean_p += dp / n;
  mean_d += dd / n;
  cpp += dp * (p - mean_p);
  cpd += dp * (d - mean_d);
  cdd += dd * (d - mean_d);
  sum_dd += d * d;
  max_abs = std::max(max_abs, std::abs(d));
}

ErrorStats OutputCheck::Fit::stats() const {
  ErrorStats s{.count = n, .max_abs = max_abs};
  if (n == 0) {
    return s;
  }
  const bool sloped = cpp > 0.0;
  s.rmse = std::sqrt(sum_dd / n);
  s.rel_bias = sloped ? cpd / cpp : 0.0;
  s.abs_bias = mean_d - s.rel_bias * mean_p;
  const int64_t dof = n - (sloped ? 2 : 1);
  const double residual = dof > 0 ? std::max(0.0, cdd - s.rel_bias * cpd) / dof
                                  : std::numeric_limits<double>::infinity();
  s.abs_bias_se =
      std::sqrt(residual * (1.0 / n + (sloped ? mean_p * mean_p / cpp : 0.0)));
  s.rel_bias_se = sloped ? std::sqrt(residual / cpp) : 0.0;
  return s;
}

void OutputCheck::add(
    std::span<const float> got,
    const SampleTensor& golden,
    const SampleTensor& precise,
    std::span<const double> allowance) {
  const std::vector<float> want = to_float(golden);
  const std::vector<float> ref = to_float(precise);
  if (got.size() != want.size() || got.size() != ref.size() ||
      (!allowance.empty() && allowance.size() != got.size())) {
    throw std::invalid_argument(
        "OutputCheck: " + std::to_string(got.size()) + " values against " +
        std::to_string(want.size()) + " golden, " + std::to_string(ref.size()) +
        " precise and " + std::to_string(allowance.size()) + " allowances");
  }
  for (size_t i = 0; i < got.size(); ++i) {
    const double g = want[i];
    const double slack = allowance.empty() ? 0.0 : allowance[i];
    const double e = std::isfinite(got[i]) && std::isfinite(g)
        ? got[i] - std::clamp(got[i] - g, -slack, slack)
        : got[i];
    const double p = ref[i];
    if (!std::isfinite(e) || !std::isfinite(g) || !std::isfinite(p)) {
      if (!same(e, g)) {
        ++nonfinite_mismatches_;
      }
      continue;
    }
    engine_.add(p, e - p);
    eager_.add(p, g - p);
    sum_pp_ += p * p;
    max_p_ = std::max(max_p_, std::abs(p));
  }
}

std::vector<std::string> OutputCheck::failures(const Policy& policy) const {
  const ErrorStats e = engine();
  const ErrorStats g = eager();
  const double rms_p = e.count > 0 ? std::sqrt(sum_pp_ / e.count) : 0.0;
  std::vector<std::string> out;
  const auto bound = [&](const char* name, double value, double limit) {
    if (!(value <= limit)) {
      out.push_back(std::string(name) + ": " + g3(value) + " > " + g3(limit));
    }
  };
  bound("rmse", e.rmse, policy.rmse_ratio * g.rmse + policy.rtol * rms_p);
  bound(
      "max_abs",
      e.max_abs,
      policy.max_ratio * g.max_abs + policy.rtol * max_p_);
  bound(
      "abs_bias",
      std::abs(e.abs_bias),
      policy.rmse_ratio * std::abs(g.abs_bias) + policy.rtol * rms_p +
          policy.sigmas * e.abs_bias_se);
  bound(
      "rel_bias",
      std::abs(e.rel_bias),
      policy.rmse_ratio * std::abs(g.rel_bias) + policy.rtol +
          policy.sigmas * e.rel_bias_se);
  if (nonfinite_mismatches_ > 0) {
    out.push_back(
        "nonfinite: " + std::to_string(nonfinite_mismatches_) +
        " values differ from golden where a value is NaN or infinite");
  }
  return out;
}

std::string OutputCheck::summary() const {
  const ErrorStats e = engine();
  const ErrorStats g = eager();
  return "n " + std::to_string(e.count) + ", engine (eager): rmse " +
      g3(e.rmse) + " (" + g3(g.rmse) + "), max_abs " + g3(e.max_abs) + " (" +
      g3(g.max_abs) + "), abs_bias " + g3(e.abs_bias) + " +-" +
      g3(e.abs_bias_se) + " (" + g3(g.abs_bias) + "), rel_bias " +
      g3(e.rel_bias) + " +-" + g3(e.rel_bias_se) + " (" + g3(g.rel_bias) + ")";
}

} // namespace ptn::testing
