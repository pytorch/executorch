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

/**
 * A tensor class for ExecuTorch operations.
 *
 * This class encapsulates a native TensorPtr instance and provides a variety of
 * initializers and utility methods to work with tensor data.
 */
NS_SWIFT_NAME(Tensor)
__attribute__((deprecated("This API is experimental.")))
@interface ExecuTorchTensor : NSObject

/**
 * Pointer to the underlying native TensorPtr instance.
 *
 * @return A raw pointer to the native TensorPtr held by this Tensor class.
 */
@property(nonatomic, readonly) void *nativeInstance NS_SWIFT_UNAVAILABLE("");

/**
 * The data type of the tensor.
 *
 * @return An ExecuTorchDataType value representing the tensor's element type.
 */
@property(nonatomic, readonly) ExecuTorchDataType dataType;

/**
 * The shape of the tensor.
 *
 * @return An NSArray of NSNumber objects representing the size of each dimension.
 */
@property(nonatomic, readonly) NSArray<NSNumber *> *shape;

/**
 * The order of dimensions in the tensor.
 *
 * @return An NSArray of NSNumber objects representing the tensor’s dimension order.
 */
@property(nonatomic, readonly) NSArray<NSNumber *> *dimensionOrder;

/**
 * The strides of the tensor.
 *
 * @return An NSArray of NSNumber objects representing the step sizes for each dimension.
 */
@property(nonatomic, readonly) NSArray<NSNumber *> *strides;

/**
 * The dynamism of the tensor's shape.
 *
 * @return An ExecuTorchShapeDynamism value indicating whether the tensor shape is static or dynamic.
 */
@property(nonatomic, readonly) ExecuTorchShapeDynamism shapeDynamism;

/**
 * The total number of elements in the tensor.
 *
 * @return An NSInteger representing the total element count.
 */
@property(nonatomic, readonly) NSInteger count;

/**
 * Initializes a tensor with a native TensorPtr instance.
 *
 * @param nativeInstance A pointer to a native TensorPtr instance.
 * @return An initialized ExecuTorchTensor instance.
 */
- (instancetype)initWithNativeInstance:(void *)nativeInstance
    NS_DESIGNATED_INITIALIZER NS_SWIFT_UNAVAILABLE("");

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
