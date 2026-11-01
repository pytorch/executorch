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
#include <limits>
#include <stdexcept>

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
    default:
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
    const SampleTensor& precise) {
  const std::vector<float> want = to_float(golden);
  const std::vector<float> ref = to_float(precise);
  if (got.size() != want.size() || got.size() != ref.size()) {
    throw std::invalid_argument(
        "OutputCheck: " + std::to_string(got.size()) + " values against " +
        std::to_string(want.size()) + " golden and " +
        std::to_string(ref.size()) + " precise");
  }
  for (size_t i = 0; i < got.size(); ++i) {
    const double e = got[i];
    const double g = want[i];
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
