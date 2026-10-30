// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <executorch/backends/native/runtime/deserialize/Samples.h>

namespace ptn::testing {

// Values of a Float, Half, BFloat16 or Double sample tensor. Throws
// std::invalid_argument for any other dtype.
std::vector<float> to_float(const SampleTensor& t);

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
  // Throws std::invalid_argument when the element counts differ.
  void add(
      std::span<const float> got,
      const SampleTensor& golden,
      const SampleTensor& precise);

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
