// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/examples/runner_utils/runner_utils.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace ptn::runner_utils {

std::vector<float> read_floats(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("cannot open " + path);
  }
  const std::streamsize size = file.tellg();
  if (size < 0 || size % static_cast<std::streamsize>(sizeof(float)) != 0) {
    throw std::runtime_error("not a whole number of floats: " + path);
  }
  file.seekg(0, std::ios::beg);
  std::vector<float> out(static_cast<size_t>(size) / sizeof(float));
  // cppcheck-suppress invalidPointerCast
  if (size > 0 && !file.read(reinterpret_cast<char*>(out.data()), size)) {
    throw std::runtime_error("cannot read " + path);
  }
  return out;
}

std::vector<std::string> read_lines(const std::string& path) {
  std::vector<std::string> lines;
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("cannot open " + path);
  }
  std::string line;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }
  return lines;
}

int64_t numel_of(const std::vector<int64_t>& sizes) {
  return std::accumulate(
      sizes.begin(), sizes.end(), int64_t{1}, std::multiplies<int64_t>());
}

std::string shape_str(const std::vector<int64_t>& sizes) {
  std::string out = "[";
  for (size_t i = 0; i < sizes.size(); ++i) {
    out += (i == 0 ? "" : ", ") + std::to_string(sizes[i]);
  }
  return out + "]";
}

std::vector<int> top_k(const std::vector<float>& values, int k) {
  std::vector<int> order(values.size());
  std::iota(order.begin(), order.end(), 0);
  const int count = std::clamp<int>(k, 0, static_cast<int>(values.size()));
  std::partial_sort(
      order.begin(), order.begin() + count, order.end(), [&](int a, int b) {
        return values[a] > values[b];
      });
  order.resize(count);
  return order;
}

bool compare_floats(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    double rtol,
    double atol) {
  if (!std::isfinite(rtol) || !std::isfinite(atol) || rtol < 0 || atol < 0) {
    std::fprintf(
        stderr,
        "error: tolerances must be finite and non-negative (rtol=%g, "
        "atol=%g)\n",
        rtol,
        atol);
    return false;
  }
  if (actual.size() != expected.size()) {
    std::fprintf(
        stderr,
        "error: reference has %zu values, model produced %zu\n",
        expected.size(),
        actual.size());
    return false;
  }
  size_t failures = 0;
  size_t worst_at = 0;
  double worst_excess = 0.0;
  double max_abs_err = 0.0;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
      ++failures;
      if (!std::isinf(worst_excess)) {
        worst_at = i;
        worst_excess = std::numeric_limits<double>::infinity();
      }
      max_abs_err = std::numeric_limits<double>::infinity();
      continue;
    }
    const double abs_err = std::fabs(
        static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
    const double budget =
        atol + rtol * std::fabs(static_cast<double>(expected[i]));
    max_abs_err = std::max(max_abs_err, abs_err);
    if (abs_err > budget) {
      ++failures;
      if (abs_err - budget > worst_excess) {
        worst_excess = abs_err - budget;
        worst_at = i;
      }
    }
  }
  std::printf(
      "compare:   %zu values, max abs err %.6g, rtol %g atol %g\n",
      actual.size(),
      max_abs_err,
      rtol,
      atol);
  if (failures == 0) {
    return true;
  }
  std::printf(
      "  %zu value(s) outside tolerance; worst at %zu: got %.6g, want %.6g\n",
      failures,
      worst_at,
      static_cast<double>(actual[worst_at]),
      static_cast<double>(expected[worst_at]));
  return false;
}

} // namespace ptn::runner_utils
