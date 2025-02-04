/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <unordered_map>

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

const std::unordered_map<ScalarType, int> scalar_type_to_java_dtype = {
    {ScalarType::Byte, kTensorDTypeUInt8},
    {ScalarType::Char, kTensorDTypeInt8},
    {ScalarType::Short, kTensorDTypeInt16},
    {ScalarType::Int, kTensorDTypeInt32},
    {ScalarType::Long, kTensorDTypeInt64},
    {ScalarType::Half, kTensorDTypeHalf},
    {ScalarType::Float, kTensorDTypeFloat},
    {ScalarType::Double, kTensorDTypeDouble},
    // These types are not supported yet
    // {ScalarType::ComplexHalf, kTensorDTypeComplexHalf},
    // {ScalarType::ComplexFloat, kTensorDTypeComplexFloat},
    // {ScalarType::ComplexDouble, kTensorDTypeComplexDouble},
    {ScalarType::Bool, kTensorDTypeBool},
    {ScalarType::QInt8, kTensorDTypeQint8},
    {ScalarType::QUInt8, kTensorDTypeQuint8},
    {ScalarType::QInt32, kTensorDTypeQint32},
    {ScalarType::BFloat16, kTensorDTypeBFloat16},
    {ScalarType::QUInt4x2, kTensorDTypeQuint4x2},
    {ScalarType::QUInt2x4, kTensorDTypeQuint2x4},
    {ScalarType::Bits1x8, kTensorDTypeBits1x8},
    {ScalarType::Bits2x4, kTensorDTypeBits2x4},
    {ScalarType::Bits4x2, kTensorDTypeBits4x2},
    {ScalarType::Bits8, kTensorDTypeBits8},
    {ScalarType::Bits16, kTensorDTypeBits16},
};

const std::unordered_map<int, ScalarType> java_dtype_to_scalar_type = {
    {kTensorDTypeUInt8, ScalarType::Byte},
    {kTensorDTypeInt8, ScalarType::Char},
    {kTensorDTypeInt16, ScalarType::Short},
    {kTensorDTypeInt32, ScalarType::Int},
    {kTensorDTypeInt64, ScalarType::Long},
    {kTensorDTypeHalf, ScalarType::Half},
    {kTensorDTypeFloat, ScalarType::Float},
    {kTensorDTypeDouble, ScalarType::Double},
    // These types are not supported yet
    // {kTensorDTypeComplexHalf, ScalarType::ComplexHalf},
    // {kTensorDTypeComplexFloat, ScalarType::ComplexFloat},
    // {kTensorDTypeComplexDouble, ScalarType::ComplexDouble},
    {kTensorDTypeBool, ScalarType::Bool},
    {kTensorDTypeQint8, ScalarType::QInt8},
    {kTensorDTypeQuint8, ScalarType::QUInt8},
    {kTensorDTypeQint32, ScalarType::QInt32},
    {kTensorDTypeBFloat16, ScalarType::BFloat16},
    {kTensorDTypeQuint4x2, ScalarType::QUInt4x2},
    {kTensorDTypeQuint2x4, ScalarType::QUInt2x4},
    {kTensorDTypeBits1x8, ScalarType::Bits1x8},
    {kTensorDTypeBits2x4, ScalarType::Bits2x4},
    {kTensorDTypeBits4x2, ScalarType::Bits4x2},
    {kTensorDTypeBits8, ScalarType::Bits8},
    {kTensorDTypeBits16, ScalarType::Bits16},
};

} // namespace executorch::extension
