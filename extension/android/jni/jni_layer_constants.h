/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace executorch::extension {

constexpr static int kTensorDTypeUInt8 = 0;
constexpr static int kTensorDTypeInt8 = 1;
constexpr static int kTensorDTypeInt16 = 2;
constexpr static int kTensorDTypeInt32 = 3;
constexpr static int kTensorDTypeInt64 = 4;
constexpr static int kTensorDTypeHalf = 5;
constexpr static int kTensorDTypeFloat = 6;
constexpr static int kTensorDTypeDouble = 7;
// These types are not supported yet
// constexpr static int kTensorDTypeComplexHalf = 8;
// constexpr static int kTensorDTypeComplexFloat = 9;
// constexpr static int kTensorDTypeComplexDouble = 10;
constexpr static int kTensorDTypeBool = 11;
constexpr static int kTensorDTypeQint8 = 12;
constexpr static int kTensorDTypeQuint8 = 13;
constexpr static int kTensorDTypeQint32 = 14;
constexpr static int kTensorDTypeBFloat16 = 15;
constexpr static int kTensorDTypeQuint4x2 = 16;
constexpr static int kTensorDTypeQuint2x4 = 17;
constexpr static int kTensorDTypeBits1x8 = 18;
constexpr static int kTensorDTypeBits2x4 = 19;
constexpr static int kTensorDTypeBits4x2 = 20;
constexpr static int kTensorDTypeBits8 = 21;
constexpr static int kTensorDTypeBits16 = 22;

using executorch::aten::ScalarType;

// Returns the Java dtype code for a ScalarType, or -1 if unsupported.
constexpr int scalar_type_to_java_dtype(ScalarType scalar_type) {
  switch (scalar_type) {
    case ScalarType::Byte:
      return kTensorDTypeUInt8;
    case ScalarType::Char:
      return kTensorDTypeInt8;
    case ScalarType::Short:
      return kTensorDTypeInt16;
    case ScalarType::Int:
      return kTensorDTypeInt32;
    case ScalarType::Long:
      return kTensorDTypeInt64;
    case ScalarType::Half:
      return kTensorDTypeHalf;
    case ScalarType::Float:
      return kTensorDTypeFloat;
    case ScalarType::Double:
      return kTensorDTypeDouble;
    // These types are not supported yet
    // case ScalarType::ComplexHalf:
    // case ScalarType::ComplexFloat:
    // case ScalarType::ComplexDouble:
    case ScalarType::Bool:
      return kTensorDTypeBool;
    case ScalarType::QInt8:
      return kTensorDTypeQint8;
    case ScalarType::QUInt8:
      return kTensorDTypeQuint8;
    case ScalarType::QInt32:
      return kTensorDTypeQint32;
    case ScalarType::BFloat16:
      return kTensorDTypeBFloat16;
    case ScalarType::QUInt4x2:
      return kTensorDTypeQuint4x2;
    case ScalarType::QUInt2x4:
      return kTensorDTypeQuint2x4;
    case ScalarType::Bits1x8:
      return kTensorDTypeBits1x8;
    case ScalarType::Bits2x4:
      return kTensorDTypeBits2x4;
    case ScalarType::Bits4x2:
      return kTensorDTypeBits4x2;
    case ScalarType::Bits8:
      return kTensorDTypeBits8;
    case ScalarType::Bits16:
      return kTensorDTypeBits16;
    default:
      return -1;
  }
}

// Returns the ScalarType for a Java dtype code, or ScalarType::Undefined if
// unsupported.
constexpr ScalarType java_dtype_to_scalar_type(int java_dtype) {
  switch (java_dtype) {
    case kTensorDTypeUInt8:
      return ScalarType::Byte;
    case kTensorDTypeInt8:
      return ScalarType::Char;
    case kTensorDTypeInt16:
      return ScalarType::Short;
    case kTensorDTypeInt32:
      return ScalarType::Int;
    case kTensorDTypeInt64:
      return ScalarType::Long;
    case kTensorDTypeHalf:
      return ScalarType::Half;
    case kTensorDTypeFloat:
      return ScalarType::Float;
    case kTensorDTypeDouble:
      return ScalarType::Double;
    // These types are not supported yet
    // case kTensorDTypeComplexHalf:
    // case kTensorDTypeComplexFloat:
    // case kTensorDTypeComplexDouble:
    case kTensorDTypeBool:
      return ScalarType::Bool;
    case kTensorDTypeQint8:
      return ScalarType::QInt8;
    case kTensorDTypeQuint8:
      return ScalarType::QUInt8;
    case kTensorDTypeQint32:
      return ScalarType::QInt32;
    case kTensorDTypeBFloat16:
      return ScalarType::BFloat16;
    case kTensorDTypeQuint4x2:
      return ScalarType::QUInt4x2;
    case kTensorDTypeQuint2x4:
      return ScalarType::QUInt2x4;
    case kTensorDTypeBits1x8:
      return ScalarType::Bits1x8;
    case kTensorDTypeBits2x4:
      return ScalarType::Bits2x4;
    case kTensorDTypeBits4x2:
      return ScalarType::Bits4x2;
    case kTensorDTypeBits8:
      return ScalarType::Bits8;
    case kTensorDTypeBits16:
      return ScalarType::Bits16;
    default:
      return ScalarType::Undefined;
  }
}

} // namespace executorch::extension
