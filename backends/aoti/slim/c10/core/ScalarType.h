/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>

#include <executorch/runtime/core/portable_type/bfloat16.h>
#include <executorch/runtime/platform/assert.h>

namespace executorch::backends::aoti::slim::c10 {

// Import BFloat16 from ExecuTorch's portable_type
using BFloat16 = ::executorch::runtime::etensor::BFloat16;

/// Enum representing the scalar type (dtype) of tensor elements.
/// Note: Enum values must match PyTorch's c10::ScalarType for compatibility.
enum class ScalarType : int8_t {
  // Byte = 0,     // uint8_t - not currently needed
  Char = 1, // int8_t
  Short = 2, // int16_t
  Int = 3, // int32_t
  Long = 4, // int64_t
  // Half = 5,     // float16 - not currently needed
  Float = 6, // float
  // Double = 7,   // double - not currently needed
  // ComplexHalf = 8,
  // ComplexFloat = 9,
  // ComplexDouble = 10,
  Bool = 11, // bool
  // QInt8 = 12,
  // QUInt8 = 13,
  // QInt32 = 14,
  BFloat16 = 15, // bfloat16
  Undefined = -1,
};

// Type alias constants for convenience
constexpr ScalarType kChar = ScalarType::Char;
constexpr ScalarType kShort = ScalarType::Short;
constexpr ScalarType kInt = ScalarType::Int;
constexpr ScalarType kLong = ScalarType::Long;
constexpr ScalarType kFloat = ScalarType::Float;
constexpr ScalarType kBool = ScalarType::Bool;
constexpr ScalarType kBFloat16 = ScalarType::BFloat16;

/// Returns the size in bytes of a single element of the given scalar type.
/// @param t The scalar type.
/// @return The size in bytes of a single element.
inline size_t elementSize(ScalarType t) {
  switch (t) {
    case ScalarType::Char:
      return sizeof(int8_t);
    case ScalarType::Short:
      return sizeof(int16_t);
    case ScalarType::Int:
      return sizeof(int32_t);
    case ScalarType::Long:
      return sizeof(int64_t);
    case ScalarType::Float:
      return sizeof(float);
    case ScalarType::Bool:
      return sizeof(bool);
    case ScalarType::BFloat16:
      return sizeof(BFloat16);
    default:
      ET_CHECK_MSG(false, "Unknown ScalarType: %d", static_cast<int>(t));
  }
}

/// Returns the name of the scalar type as a string.
/// @param t The scalar type.
/// @return The name of the scalar type.
inline const char* toString(ScalarType t) {
  switch (t) {
    case ScalarType::Char:
      return "Char";
    case ScalarType::Short:
      return "Short";
    case ScalarType::Int:
      return "Int";
    case ScalarType::Long:
      return "Long";
    case ScalarType::Float:
      return "Float";
    case ScalarType::Bool:
      return "Bool";
    case ScalarType::BFloat16:
      return "BFloat16";
    case ScalarType::Undefined:
      return "Undefined";
    default:
      return "UNKNOWN_SCALAR";
  }
}

/// Checks if the scalar type is a floating point type.
/// @param t The scalar type to check.
/// @return true if the scalar type is floating point, false otherwise.
inline bool isFloatingType(ScalarType t) {
  return t == ScalarType::Float || t == ScalarType::BFloat16;
}

/// Checks if the scalar type is an integral type (including bool optionally).
/// @param t The scalar type to check.
/// @param includeBool Whether to consider Bool as integral.
/// @return true if the scalar type is integral, false otherwise.
inline bool isIntegralType(ScalarType t, bool includeBool) {
  switch (t) {
    case ScalarType::Char:
    case ScalarType::Short:
    case ScalarType::Int:
    case ScalarType::Long:
      return true;
    case ScalarType::Bool:
      return includeBool;
    default:
      return false;
  }
}

/// Checks if the scalar type is a boolean type.
/// @param t The scalar type to check.
/// @return true if the scalar type is Bool, false otherwise.
inline bool isBoolType(ScalarType t) {
  return t == ScalarType::Bool;
}

inline std::ostream& operator<<(std::ostream& stream, ScalarType scalar_type) {
  return stream << toString(scalar_type);
}

} // namespace executorch::backends::aoti::slim::c10
