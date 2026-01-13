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

#include <executorch/runtime/platform/assert.h>

namespace executorch::backends::aoti::slim::c10 {

/// Enum representing the scalar type (dtype) of tensor elements.
/// Note: Enum values must match PyTorch's c10::ScalarType for compatibility.
enum class ScalarType : int8_t {
  // Byte = 0,
  // Char = 1,
  // Short = 2,
  // Int = 3,
  // Long = 4,
  Float = 6,
  // Bool = 11,
  // BFloat16 = 15,
  Undefined = -1,
  NumOptions = 7,
};

/// Constant for Float scalar type.
constexpr ScalarType kFloat = ScalarType::Float;

/// Returns the size in bytes of a single element of the given scalar type.
/// @param t The scalar type.
/// @return The size in bytes of a single element.
inline size_t elementSize(ScalarType t) {
  switch (t) {
    case ScalarType::Float:
      return sizeof(float);
    default:
      ET_CHECK_MSG(false, "Unknown ScalarType: %d", static_cast<int>(t));
  }
}

/// Returns the name of the scalar type as a string.
/// @param t The scalar type.
/// @return The name of the scalar type.
inline const char* toString(ScalarType t) {
  switch (t) {
    case ScalarType::Float:
      return "Float";
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
  return t == ScalarType::Float;
}

/// Checks if the scalar type is an integral type (including bool).
/// @param t The scalar type to check.
/// @param includeBool Whether to consider Bool as integral.
/// @return true if the scalar type is integral, false otherwise.
inline bool isIntegralType(ScalarType t, bool /*includeBool*/) {
  (void)t;
  return false;
}

inline std::ostream& operator<<(std::ostream& stream, ScalarType scalar_type) {
  return stream << toString(scalar_type);
}

} // namespace executorch::backends::aoti::slim::c10
