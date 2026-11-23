// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <executorch/backends/native/runtime/deserialize/Samples.h>

namespace ptn::testing {

// Values of a Float, Half, BFloat16 or Double sample tensor. Throws
// std::invalid_argument for any other dtype.
std::vector<float> to_float(const SampleTensor& t);

// ggml's CUDA and HIP kernels for Q4_1 and Q5_1 linears of up to 8 activation
// rows quantize each 32-value block of a row as Q8_1: d8 = amax / 127,
// q8 = round(x / d8) and s = d8 * sum(q8), d8 and s stored in fp16. A block of
// weights d * q + m then adds sum(q * q8) * fp16(d * d8) + fp16(m * s), each
// fp16 product off its exact value by at most half an fp16 ulp: 2^-11 of it
// when normal, 2^-25 below 2^-14, which also covers the products under 2^-25
// that flush to zero. An output's error from these products is thus at most
//   sum over its blocks of |sum(q * q8)| * ulp(d * d8) / 2 + ulp(m * s) / 2.
// Error e in x, from an earlier linear, adds |w| * e and may move each q8 by
// e / d8 + 1, which the bound takes at its largest; it always allows one step,
// for division that rounds differently.
struct BlockScaleLinear {
  // [rows, out] outputs with exact scale products, without bias.
  std::vector<double> outputs;
  // [rows, out] bounds on the error of the GPU's fp16 products in them.
  std::vector<double> bounds;
};

// `codec` is gguf:q4_1 or gguf:q5_1, `weight` the [out, in] blocks, `x` the
// [rows, in] activations and `x_error` a bound on each one's error, or empty.
// ggml takes these kernels for matmuls of up to 8 rows in their second-to-last
// dim; others have exact products, so `vector_kernel` false allows only the
// error of x. Throws std::invalid_argument on other codecs or sizes.
BlockScaleLinear block_scale_linear(
    std::string_view codec,
    std::span<const uint8_t> weight,
    int64_t out,
    int64_t in,
    std::span<const double> x,
    std::span<const double> x_error,
    bool vector_kernel);

// Error d = x - precise, fitted by least squares as
// d = abs_bias + rel_bias * precise; the _se fields are the fit's standard
// errors.
struct ErrorStats {
  int64_t count = 0;
  double rmse = 0.0;
  double max_abs = 0.0;
  double abs_bias = 0.0;
  double abs_bias_se = 0.0;
  double rel_bias = 0.0;
  double rel_bias_se = 0.0;
};

// Bounds on an engine's error statistics: eager's times rmse_ratio (max_ratio
// for max_abs), plus rtol times the precise outputs' RMS (their max for
// max_abs, 1 for rel_bias). The fitted biases may also deviate by `sigmas`
// standard errors.
struct Policy {
  double rmse_ratio = 3.0;
  double max_ratio = 4.0;
  double rtol = 1e-5;
  double sigmas = 6.0;
};

// One output of a method, accumulated over its samples. Statistics cover the
// positions where the engine, golden and precise values are all finite;
// elsewhere the engine must equal golden, NaN matching NaN.
class OutputCheck {
 public:
  // Throws std::invalid_argument when the element counts differ. A nonempty
  // `allowance` moves each engine value toward golden by up to its entry
  // before the statistics see it.
  void add(
      std::span<const float> got,
      const SampleTensor& golden,
      const SampleTensor& precise,
      std::span<const double> allowance = {});

  ErrorStats engine() const {
    return engine_.stats();
  }
  ErrorStats eager() const {
    return eager_.stats();
  }

  // One "<rule>: <detail>" message per violated rule, among rmse, max_abs,
  // abs_bias, rel_bias and nonfinite; empty when the engine passes.
  std::vector<std::string> failures(const Policy& policy) const;

  std::string summary() const;

 private:
  // Streaming least-squares fit of d against p.
  struct Fit {
    int64_t n = 0;
    double mean_p = 0.0;
    double mean_d = 0.0;
    double cpp = 0.0;
    double cpd = 0.0;
    double cdd = 0.0;
    double sum_dd = 0.0;
    double max_abs = 0.0;

    void add(double p, double d);
    ErrorStats stats() const;
  };

  Fit engine_;
  Fit eager_;
  double sum_pp_ = 0.0;
  double max_p_ = 0.0;
  int64_t nonfinite_mismatches_ = 0;
};

} // namespace ptn::testing
