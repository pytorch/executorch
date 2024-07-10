/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// NOTE: This is a modified excerpt of
//  https://github.com/microsoft/ArchProbe/blob/main/include/stats.hpp
// MIT-licensed by Rendong Liang.

// Tools for statistics.
// @PENGUINLIONG
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include "log.h"

namespace gpuinfo {

namespace stats {

template <typename T>
class MinStats {
  T mn_ = std::numeric_limits<T>::max();

 public:
  typedef T value_t;

  // Returns true if the value has been updated.
  bool push(T value) {
    if (mn_ > value) {
      mn_ = value;
      return true;
    } else {
      return false;
    }
  }
  inline bool has_value() const {
    return mn_ != std::numeric_limits<T>::max();
  }
  operator T() const {
    if (!has_value()) {
      gpuinfo::log::warn("`MinStats` has not collected any data yet");
    }
    return mn_;
  }
  friend std::ostream& operator<<(std::ostream& out, const MinStats<T>& x) {
    out << (T)(x);
    return out;
  }
};
template <typename T>
class MaxStats {
  T mx_ = -std::numeric_limits<T>::max();

 public:
  typedef T value_t;

  // Returns true if the value has been updated.
  bool push(T value) {
    if (mx_ < value) {
      mx_ = value;
      return true;
    } else {
      return false;
    }
  }
  inline bool has_value() const {
    return mx_ != -std::numeric_limits<T>::max();
  }
  operator T() const {
    if (!has_value()) {
      gpuinfo::log::warn("`MaxStats` has not collected any data yet");
    }
    return mx_;
  }
  friend std::ostream& operator<<(std::ostream& out, const MaxStats<T>& x) {
    out << (T)(x);
    return out;
  }
};
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
    if (!has_value()) {
      gpuinfo::log::warn("`AvgStats` has not collected any data yet");
    }
    return sum_ / n_;
  }
  friend std::ostream& operator<<(std::ostream& out, const AvgStats<T>& x) {
    out << (T)(x);
    return out;
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
    if (!has_value()) {
      gpuinfo::log::warn("`NTapStats` has not collected any data yet");
    }

    double out = 0.0;
    for (double x : hist_) {
      out += x;
    }
    out /= NTap;
    return out;
  }
  friend std::ostream& operator<<(
      std::ostream& out,
      const NTapAvgStats<T, NTap>& x) {
    out << (T)(x);
    return out;
  }
};
template <typename T>
class StdStats {
  AvgStats<T> avg_{};
  std::vector<T> values_{};

 public:
  typedef T value_t;

  void push(T value) {
    avg_.push(value);
    values_.push_back(value);
  }
  inline bool has_value() const {
    return avg_.has_value();
  }
  operator T() const {
    if (!has_value()) {
      gpuinfo::log::warn("`StdStats` has not collected any data yet");
    }
    T avg = avg_;
    T sqr_sum = 0;
    for (auto value : values_) {
      auto temp = value - avg;
      sqr_sum += temp * temp;
    }
    return std::sqrt(sqr_sum / values_.size());
  }
  friend std::ostream& operator<<(std::ostream& out, const StdStats<T>& x) {
    out << (T)(x);
    return out;
  }
};
template <typename T>
class MedianStats {
  std::vector<T> values_{};

 public:
  typedef T value_t;

  void push(T value) {
    values_.push_back(value);
  }
  inline bool has_value() const {
    return !values_.empty();
  }
  operator T() {
    if (!has_value()) {
      gpuinfo::log::warn("`MedianStats` has not collected any data yet");
    }
    std::sort(values_.begin(), values_.end());
    size_t imid = values_.size() / 2;
    if (values_.size() & 1) {
      return values_[imid];
    } else {
      return (values_[imid] + values_[imid + 1]) / 2;
    }
  }
  friend std::ostream& operator<<(std::ostream& out, const MedianStats<T>& x) {
    out << (T)(x);
    return out;
  }
};

template <typename TStats>
class GeomDeltaStats {
  TStats stats_{};
  bool has_ratio_ = false;
  typename TStats::value_t ratio_{};

 public:
  typedef typename TStats::value_t value_t;

  void push(value_t value) {
    if (stats_.has_value()) {
      ratio_ = value / (value_t)stats_;
      has_ratio_ = true;
    }
    stats_.push(value);
  }
  inline bool has_value() const {
    return has_ratio_;
  }
  operator value_t() const {
    if (!has_value()) {
      gpuinfo::log::warn("`GeomDeltaStats` has not collected enough data yet");
    }
    return ratio_;
  }
  friend std::ostream& operator<<(
      std::ostream& out,
      const GeomDeltaStats<TStats>& x) {
    if (x.has_value()) {
      out << (typename TStats::value_t)(x.ratio_);
    }
    return out;
  }
};
template <typename TStats>
class ArithDeltaStats {
  TStats stats_{};
  bool has_delta_ = false;
  typename TStats::value_t delta_{};

 public:
  typedef typename TStats::value_t value_t;

  void push(value_t value) {
    if (stats_.has_value()) {
      delta_ = value - (value_t)stats_;
      has_delta_ = true;
    }
    stats_.push(value);
  }
  inline bool has_value() const {
    return has_delta_;
  }
  operator value_t() const {
    if (!has_value()) {
      gpuinfo::log::warn("`ArithDeltaStats` has not collected enough data yet");
    }
    return delta_;
  }
  friend std::ostream& operator<<(
      std::ostream& out,
      const ArithDeltaStats<TStats>& x) {
    if (x.has_value()) {
      out << (typename TStats::value_t)(x.delta_);
    }
    return out;
  }
};

} // namespace stats

} // namespace gpuinfo
