/*
 * Portions (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Code sourced from
 * https://github.com/microsoft/ArchProbe/blob/main/include/stats.hpp with the
 * following MIT license
 *
 * MIT License
 *
 * Copyright (c) Microsoft Corporation.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE
 */

#pragma once
#include <array>
#include <cstdint>

template <typename T>
class AvgStats {
  T sum_ = 0;
  uint64_t n_ = 0;

 public:
  typedef T value_t;

  void push(T value) {
    sum_ += value;
    n_ += 1;
  }
  inline bool has_value() const {
    return n_ != 0;
  }
  operator T() const {
    return sum_ / n_;
  }
};

template <typename T, size_t NTap>
class NTapAvgStats {
  std::array<double, NTap> hist_;
  size_t cur_idx_;
  bool ready_;

 public:
  typedef T value_t;

  void push(T value) {
    hist_[cur_idx_++] = value;
    if (cur_idx_ >= NTap) {
      cur_idx_ = 0;
      ready_ = true;
    }
  }
  inline bool has_value() const {
    return ready_;
  }
  operator T() const {
    double out = 0.0;
    for (double x : hist_) {
      out += x;
    }
    out /= NTap;
    return out;
  }
};

template <uint32_t NTap>
struct DtJumpFinder {
 private:
  NTapAvgStats<double, NTap> time_avg_;
  AvgStats<double> dtime_avg_;
  double compensation_;
  double threshold_;

 public:
  // Compensation is a tiny additive to give on delta time so that the algorithm
  // works smoothly when a sequence of identical timing is ingested, which is
  // pretty common in our tests. Threshold is simply how many times the new
  // delta has to be to be recognized as a deviation.
  DtJumpFinder(double compensation = 0.01, double threshold = 10)
      : time_avg_(),
        dtime_avg_(),
        compensation_(compensation),
        threshold_(threshold) {}

  // Returns true if the delta time regarding to the last data point seems
  // normal; returns false if it seems the new data point is too much away from
  // the historical records.
  bool push(double time) {
    if (time_avg_.has_value()) {
      double dtime = std::abs(time - time_avg_) + (compensation_ * time_avg_);
      if (dtime_avg_.has_value()) {
        double ddtime = std::abs(dtime - dtime_avg_);
        if (ddtime > threshold_ * dtime_avg_) {
          return true;
        }
      }
      dtime_avg_.push(dtime);
    }
    time_avg_.push(time);
    return false;
  }

  double dtime_avg() const {
    return dtime_avg_;
  }
  double compensate_time() const {
    return compensation_ * time_avg_;
  }
};
