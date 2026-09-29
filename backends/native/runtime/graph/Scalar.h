// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <variant>

namespace ptn {

// A concrete scalar value, analogous to c10::Scalar: one alternative per
// domain — integral, floating point, boolean — each stored at its widest type.
// A narrower value is stored exactly, and to<T>() narrows it back on read.
// There is no "none" state — the graph's Value owns that.
class Scalar {
 private:
  std::variant<int64_t, double, bool> value_ = int64_t{0};

 public:
  constexpr Scalar() = default;
  // Implicit by design (ergonomic: `Scalar s = 5;`). The int overload
  // disambiguates `Scalar(5)` — without it int -> {int64_t,double,bool} is an
  // ambiguous conversion.
  // cppcheck-suppress-begin noExplicitConstructor
  /* implicit */ constexpr Scalar(int v) : value_(static_cast<int64_t>(v)) {}
  /* implicit */ constexpr Scalar(int64_t v) : value_(v) {}
  /* implicit */ constexpr Scalar(double v) : value_(v) {}
  /* implicit */ constexpr Scalar(bool v) : value_(v) {}
  // cppcheck-suppress-end noExplicitConstructor
  // Every pointer converts to bool, so without this `Scalar s = some_ptr;`
  // would quietly yield a Bool.
  template <typename T>
  Scalar(T*) = delete;

  constexpr bool is_int() const {
    return std::holds_alternative<int64_t>(value_);
  }
  constexpr bool is_double() const {
    return std::holds_alternative<double>(value_);
  }
  constexpr bool is_bool() const {
    return std::holds_alternative<bool>(value_);
  }

  // Strict accessors: throw std::runtime_error unless that alternative is live.
  int64_t to_int() const;
  double to_double() const;
  bool to_bool() const;

  // Promoting read: static_cast whichever alternative is live to T, like
  // c10::Scalar::to<T>().
  template <typename T>
  constexpr T to() const {
    return std::visit([](auto v) { return static_cast<T>(v); }, value_);
  }
};

} // namespace ptn
