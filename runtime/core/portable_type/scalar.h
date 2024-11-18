/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/portable_type/bfloat16.h>
#include <executorch/runtime/core/portable_type/half.h>
#include <executorch/runtime/core/tag.h>
#include <executorch/runtime/platform/assert.h>

#include <cstdint>
#include <type_traits>

namespace executorch {
namespace runtime {
namespace etensor {

/**
 * Represents a scalar value.
 *
 * The API is a source-compatible subset of c10::Scalar, and the
 * semantics/behavior should also match the c10 version.
 */
class Scalar {
 public:
  Scalar() : Scalar(int64_t(0)) {}

  template <
      typename T,
      typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
  /*implicit*/ Scalar(T val) : tag(Tag::Int) {
    v.as_int = static_cast<int64_t>(val);
  }
  /*implicit*/ Scalar(bool val) : tag(Tag::Bool) {
    v.as_bool = val;
  }
  /*implicit*/ Scalar(double val) : tag(Tag::Double) {
    v.as_double = val;
  }
  /*implicit*/ Scalar(BFloat16 val) : Scalar((double)(float)val) {}
  /*implicit*/ Scalar(Half val) : Scalar((double)(float)val) {}

  /// Returns the concrete scalar value stored within.
  template <typename T>
  T to() const;

  /// Returns true if the scalar is integral, false otherwise.
  bool isIntegral(bool includeBool) const {
    return Tag::Int == tag || (includeBool && isBoolean());
  }

  /// Returns true if the scalar is a floating point, false otherwise.
  bool isFloatingPoint() const {
    return tag == Tag::Double;
  }

  /// Returns true if the scalar is a boolean, false otherwise.
  bool isBoolean() const {
    return tag == Tag::Bool;
  }

 private:
  int64_t toInt() const {
    if (isIntegral(/*includeBool=*/false)) {
      return v.as_int;
    } else if (isBoolean()) {
      return static_cast<int64_t>(v.as_bool);
    } else {
      ET_CHECK_MSG(false, "Scalar is not an int nor a Boolean.");
    }
  }

  double toFloatingPoint() const {
    ET_CHECK_MSG(isFloatingPoint(), "Scalar is not a Double.");
    return v.as_double;
  }

  double toDouble() const {
    ET_CHECK_MSG(isFloatingPoint(), "Scalar is not a Double.");
    return v.as_double;
  }

  bool toBool() const {
    ET_CHECK_MSG(isBoolean(), "Scalar is not a Boolean.");
    return v.as_bool;
  }

  Tag tag;
  union v_t {
    double as_double;
    int64_t as_int;
    bool as_bool;
    v_t() {} // default constructor
  } v;
};

#define ET_DEFINE_SCALAR_TO_METHOD(T, name) \
  template <>                               \
  inline T Scalar::to<T>() const {          \
    return to##name();                      \
  }

ET_DEFINE_SCALAR_TO_METHOD(double, Double)
ET_DEFINE_SCALAR_TO_METHOD(int64_t, Int)
ET_DEFINE_SCALAR_TO_METHOD(bool, Bool)
#undef ET_DEFINE_SCALAR_TO_METHOD

} // namespace etensor
} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::etensor::Scalar;
} // namespace executor
} // namespace torch
