/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * Enum to define the data type of a Tensor.
 * Values can be a subset, but must numerically match exactly those defined in
 * runtime/core/portable_type/scalar_type.h
 */
typedef NS_ENUM(int8_t, ExecuTorchDataType) {
  ExecuTorchDataTypeByte,
  ExecuTorchDataTypeChar,
  ExecuTorchDataTypeShort,
  ExecuTorchDataTypeInt,
  ExecuTorchDataTypeLong,
  ExecuTorchDataTypeHalf,
  ExecuTorchDataTypeFloat,
  ExecuTorchDataTypeDouble,
  ExecuTorchDataTypeComplexHalf,
  ExecuTorchDataTypeComplexFloat,
  ExecuTorchDataTypeComplexDouble,
  ExecuTorchDataTypeBool,
  ExecuTorchDataTypeQInt8,
  ExecuTorchDataTypeQUInt8,
  ExecuTorchDataTypeQInt32,
  ExecuTorchDataTypeBFloat16,
  ExecuTorchDataTypeQUInt4x2,
  ExecuTorchDataTypeQUInt2x4,
  ExecuTorchDataTypeBits1x8,
  ExecuTorchDataTypeBits2x4,
  ExecuTorchDataTypeBits4x2,
  ExecuTorchDataTypeBits8,
  ExecuTorchDataTypeBits16,
  ExecuTorchDataTypeFloat8_e5m2,
  ExecuTorchDataTypeFloat8_e4m3fn,
  ExecuTorchDataTypeFloat8_e5m2fnuz,
  ExecuTorchDataTypeFloat8_e4m3fnuz,
  ExecuTorchDataTypeUInt16,
  ExecuTorchDataTypeUInt32,
  ExecuTorchDataTypeUInt64,
  ExecuTorchDataTypeUndefined,
  ExecuTorchDataTypeNumOptions,
} NS_SWIFT_NAME(DataType);

/**
 * Enum to define the shape dynamism of a Tensor.
 * Values can be a subset, but must numerically match exactly those defined in
 * runtime/core/tensor_shape_dynamism.h
 */
typedef NS_ENUM(uint8_t, ExecuTorchShapeDynamism) {
  ExecuTorchShapeDynamismStatic,
  ExecuTorchShapeDynamismDynamicBound,
  ExecuTorchShapeDynamismDynamicUnbound,
} NS_SWIFT_NAME(ShapeDynamism);

NS_SWIFT_NAME(Tensor)
__attribute__((deprecated("This API is experimental.")))
@interface ExecuTorchTensor : NSObject

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
