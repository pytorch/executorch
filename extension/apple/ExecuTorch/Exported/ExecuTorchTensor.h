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
 * Returns the size in bytes of the specified data type.
 *
 * @param dataType An ExecuTorchDataType value representing the tensor's element type.
 * @return An NSInteger indicating the size in bytes.
 */
FOUNDATION_EXPORT
__attribute__((deprecated("This API is experimental.")))
NSInteger ExecuTorchSizeOfDataType(ExecuTorchDataType dataType)
    NS_SWIFT_NAME(size(ofDataType:));

/**
 * Computes the total number of elements in a tensor based on its shape.
 *
 * @param shape An NSArray of NSNumber objects, where each element represents a dimension size.
 * @return An NSInteger equal to the product of the sizes of all dimensions.
 */
FOUNDATION_EXPORT
__attribute__((deprecated("This API is experimental.")))
NSInteger ExecuTorchElementCountOfShape(NSArray<NSNumber *> *shape)
    NS_SWIFT_NAME(elementCount(ofShape:));

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

/**
 * Executes a block with a pointer to the tensor's immutable byte data.
 *
 * @param handler A block that receives:
 *   - a pointer to the data,
 *   - the total number of elements,
 *   - and the data type.
 */
- (void)bytesWithHandler:(void (^)(const void *pointer, NSInteger count, ExecuTorchDataType dataType))handler
    NS_SWIFT_NAME(bytes(_:));

/**
 * Executes a block with a pointer to the tensor's mutable byte data.
 *
 * @param handler A block that receives:
 *   - a mutable pointer to the data,
 *   - the total number of elements,
 *   - and the data type.
 */
- (void)mutableBytesWithHandler:(void (^)(void *pointer, NSInteger count, ExecuTorchDataType dataType))handler
    NS_SWIFT_NAME(mutableBytes(_:));

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

@end

#pragma mark - BytesNoCopy Category

@interface ExecuTorchTensor (BytesNoCopy)

/**
 * Initializes a tensor without copying the provided data.
 *
 * @param pointer A pointer to the data buffer.
 * @param shape An NSArray of NSNumber objects representing the tensor's shape.
 * @param strides An NSArray of NSNumber objects representing the tensor's strides.
 * @param dimensionOrder An NSArray of NSNumber objects indicating the order of dimensions.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value indicating whether the shape is static or dynamic.
 * @return An initialized ExecuTorchTensor instance using the provided data buffer.
 */
- (instancetype)initWithBytesNoCopy:(void *)pointer
                              shape:(NSArray<NSNumber *> *)shape
                            strides:(NSArray<NSNumber *> *)strides
                     dimensionOrder:(NSArray<NSNumber *> *)dimensionOrder
                           dataType:(ExecuTorchDataType)dataType
                      shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism;

/**
 * Initializes a tensor without copying data using dynamic bound shape (default strides and dimension order).
 *
 * @param pointer A pointer to the data buffer.
 * @param shape An NSArray of NSNumber objects representing the tensor's shape.
 * @param strides An NSArray of NSNumber objects representing the tensor's strides.
 * @param dimensionOrder An NSArray of NSNumber objects indicating the order of dimensions.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @return An initialized ExecuTorchTensor instance.
 */
- (instancetype)initWithBytesNoCopy:(void *)pointer
                              shape:(NSArray<NSNumber *> *)shape
                            strides:(NSArray<NSNumber *> *)strides
                     dimensionOrder:(NSArray<NSNumber *> *)dimensionOrder
                           dataType:(ExecuTorchDataType)dataType;

/**
 * Initializes a tensor without copying data, with an explicit shape dynamism.
 *
 * @param pointer A pointer to the data buffer.
 * @param shape An NSArray of NSNumber objects representing the tensor's shape.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value indicating the shape dynamism.
 * @return An initialized ExecuTorchTensor instance.
 */
- (instancetype)initWithBytesNoCopy:(void *)pointer
                              shape:(NSArray<NSNumber *> *)shape
                           dataType:(ExecuTorchDataType)dataType
                      shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism;

/**
 * Initializes a tensor without copying data, specifying only the shape and data type.
 *
 * @param pointer A pointer to the data buffer.
 * @param shape An NSArray of NSNumber objects representing the tensor's shape.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @return An initialized ExecuTorchTensor instance.
 */
- (instancetype)initWithBytesNoCopy:(void *)pointer
                              shape:(NSArray<NSNumber *> *)shape
                           dataType:(ExecuTorchDataType)dataType;

@end

#pragma mark - Bytes Category

@interface ExecuTorchTensor (Bytes)

/**
 * Initializes a tensor by copying bytes from the provided pointer.
 *
 * @param pointer A pointer to the source data buffer.
 * @param shape An NSArray of NSNumber objects representing the tensor's shape.
 * @param strides An NSArray of NSNumber objects representing the tensor's strides.
 * @param dimensionOrder An NSArray of NSNumber objects indicating the order of dimensions.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value indicating the shape dynamism.
 * @return An initialized ExecuTorchTensor instance with its own copy of the data.
 */
- (instancetype)initWithBytes:(const void *)pointer
                        shape:(NSArray<NSNumber *> *)shape
                      strides:(NSArray<NSNumber *> *)strides
               dimensionOrder:(NSArray<NSNumber *> *)dimensionOrder
                     dataType:(ExecuTorchDataType)dataType
                shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism;

/**
 * Initializes a tensor by copying bytes from the provided pointer with dynamic bound shape.
 *
 * @param pointer A pointer to the source data buffer.
 * @param shape An NSArray of NSNumber objects representing the tensor's shape.
 * @param strides An NSArray of NSNumber objects representing the tensor's strides.
 * @param dimensionOrder An NSArray of NSNumber objects indicating the order of dimensions.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @return An initialized ExecuTorchTensor instance with its own copy of the data.
 */
- (instancetype)initWithBytes:(const void *)pointer
                        shape:(NSArray<NSNumber *> *)shape
                      strides:(NSArray<NSNumber *> *)strides
               dimensionOrder:(NSArray<NSNumber *> *)dimensionOrder
                     dataType:(ExecuTorchDataType)dataType;

/**
 * Initializes a tensor by copying bytes from the provided pointer, specifying shape, data type, and explicit shape dynamism.
 *
 * @param pointer A pointer to the source data buffer.
 * @param shape An NSArray of NSNumber objects representing the tensor's shape.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value indicating the shape dynamism.
 * @return An initialized ExecuTorchTensor instance with its own copy of the data.
 */
- (instancetype)initWithBytes:(const void *)pointer
                        shape:(NSArray<NSNumber *> *)shape
                     dataType:(ExecuTorchDataType)dataType
                shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism;

/**
 * Initializes a tensor by copying bytes from the provided pointer, specifying only the shape and data type.
 *
 * @param pointer A pointer to the source data buffer.
 * @param shape An NSArray of NSNumber objects representing the tensor's shape.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @return An initialized ExecuTorchTensor instance with its own copy of the data.
 */
- (instancetype)initWithBytes:(const void *)pointer
                        shape:(NSArray<NSNumber *> *)shape
                     dataType:(ExecuTorchDataType)dataType;

@end

NS_ASSUME_NONNULL_END
