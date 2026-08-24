// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <string>
#include <variant>

namespace ptn {

// A concrete scalar value: an int / double / bool. Analogous to c10::Scalar.
// Trivially copyable. There is no "none" state — the graph's Value owns that.
class Scalar {
 public:
  // Which alternative is live. Pinned to the order of the variant below, so a
  // Tag is exactly an index into it.
  enum class Tag : int8_t { Int = 0, Double = 1, Bool = 2 };

 private:
  std::variant<int64_t, double, bool> value_ = int64_t{0};

 public:
  constexpr Scalar() = default;
  // Implicit by design (ergonomic: `Scalar s = 5;`). The int overload
  // disambiguates `Scalar(5)` — without it int -> {int64_t,double,bool} is an
  // ambiguous conversion. The suppression is scoped to this block so a
  // genuinely accidental implicit constructor added later is still reported.
  // cppcheck-suppress-begin noExplicitConstructor
  /* implicit */ constexpr Scalar(int v) : value_(static_cast<int64_t>(v)) {}
  /* implicit */ constexpr Scalar(int64_t v) : value_(v) {}
  /* implicit */ constexpr Scalar(double v) : value_(v) {}
  /* implicit */ constexpr Scalar(bool v) : value_(v) {}
  // cppcheck-suppress-end noExplicitConstructor
  // Every pointer converts to bool, so without this `Scalar s = some_ptr;`
  // would quietly yield a Bool. Deleted rather than made explicit so the
  // implicit numeric constructors above keep their ergonomics.
  template <typename T>
  Scalar(T*) = delete;

  constexpr Tag tag() const {
    return static_cast<Tag>(value_.index());
  }
  constexpr bool is_int() const {
    return std::holds_alternative<int64_t>(value_);
  }
  constexpr bool is_double() const {
    return std::holds_alternative<double>(value_);
  }
  constexpr bool is_bool() const {
    return std::holds_alternative<bool>(value_);
  }

  // Strict accessors: return the live alternative, throw std::runtime_error on
  // a tag mismatch.
  int64_t to_int() const;
  double to_double() const;
  bool to_bool() const;

  // Promoting read: static_cast the live alternative to T (like
  // c10::Scalar::to<T>()); works whichever tag is live.
  template <typename T>
  constexpr T to() const {
    return std::visit([](auto v) { return static_cast<T>(v); }, value_);
  }

  const char* tag_name() const;
  std::string to_string() const;
};

} // namespace ptn
