/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "test_utils.h"

#include <stdexcept>

executorch::aten::ScalarType at_scalartype_to_et_scalartype(
    at::ScalarType dtype) {
  using ScalarType = executorch::aten::ScalarType;
  switch (dtype) {
    case at::kByte:
      return ScalarType::Byte;
    case at::kChar:
      return ScalarType::Char;
    case at::kShort:
      return ScalarType::Short;
    case at::kInt:
      return ScalarType::Int;
    case at::kLong:
      return ScalarType::Long;
    case at::kHalf:
      return ScalarType::Half;
    case at::kFloat:
      return ScalarType::Float;
    case at::kDouble:
      return ScalarType::Double;
    default:
      throw std::runtime_error("Unsupported dtype");
  }
}

std::string scalar_type_name(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::kLong:
      return "c10::kLong";
    case c10::kShort:
      return "c10::kShort";
    case c10::kComplexHalf:
      return "c10::kComplexHalf";
    case c10::kComplexFloat:
      return "c10::kComplexFloat";
    case c10::kComplexDouble:
      return "c10::kComplexDouble";
    case c10::kBool:
      return "c10::kBool";
    case c10::kQInt8:
      return "c10::kQInt8";
    case c10::kQUInt8:
      return "c10::kQUInt8";
    case c10::kQInt32:
      return "c10::kQInt32";
    case c10::kBFloat16:
      return "c10::kBFloat16";
    case c10::kQUInt4x2:
      return "c10::kQUInt4x2";
    case c10::kQUInt2x4:
      return "c10::kQUInt2x4";
    case c10::kFloat:
      return "c10::kFloat";
    case c10::kHalf:
      return "c10::kHalf";
    case c10::kInt:
      return "c10::kInt";
    case c10::kChar:
      return "c10::kChar";
    case c10::kByte:
      return "c10::kByte";
    case c10::kDouble:
      return "c10::kDouble";
    case c10::kUInt16:
      return "c10::kUInt16";
    case c10::kBits16:
      return "c10::kBits16";
    default:
      return "Unknown(" + std::to_string(static_cast<int>(dtype)) + ")";
  }
}

vkcompute::vkapi::ScalarType from_at_scalartype(c10::ScalarType at_scalartype) {
  using namespace vkcompute;
  switch (at_scalartype) {
    case c10::kHalf:
      return vkapi::kHalf;
    case c10::kFloat:
      return vkapi::kFloat;
    case c10::kDouble:
      return vkapi::kDouble;
    case c10::kInt:
      return vkapi::kInt;
    case c10::kLong:
      // No support for 64-bit integers
      return vkapi::kInt;
    case c10::kChar:
      return vkapi::kChar;
    case c10::kByte:
      return vkapi::kByte;
    case c10::kShort:
      return vkapi::kShort;
    case c10::kUInt16:
      return vkapi::kUInt16;
    default:
      VK_THROW(
          "Unsupported at::ScalarType: ",
          scalar_type_name(at_scalartype),
          " (",
          static_cast<int>(at_scalartype),
          ")");
  }
}
