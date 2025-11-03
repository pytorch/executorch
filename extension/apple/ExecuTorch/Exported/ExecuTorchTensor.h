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
NSInteger ExecuTorchSizeOfDataType(ExecuTorchDataType dataType)
    NS_SWIFT_NAME(size(ofDataType:));

/**
 * Computes the total number of elements in a tensor based on its shape.
 *
 * @param shape An NSArray of NSNumber objects, where each element represents a dimension size.
 * @return An NSInteger equal to the product of the sizes of all dimensions.
 */
FOUNDATION_EXPORT
NSInteger ExecuTorchElementCountOfShape(NSArray<NSNumber *> *shape)
    NS_REFINED_FOR_SWIFT;

/**
 * A tensor class for ExecuTorch operations.
 *
 * This class encapsulates a native TensorPtr instance and provides a variety of
 * initializers and utility methods to work with tensor data.
 */
 NS_SWIFT_NAME(AnyTensor)
__attribute__((objc_subclassing_restricted))
@interface ExecuTorchTensor : NSObject<NSCopying>

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
@property(nonatomic, readonly) NSArray<NSNumber *> *shape NS_REFINED_FOR_SWIFT;

/**
 * The order of dimensions in the tensor.
 *
 * @return An NSArray of NSNumber objects representing the tensor’s dimension order.
 */
@property(nonatomic, readonly) NSArray<NSNumber *> *dimensionOrder NS_REFINED_FOR_SWIFT;

/**
 * The strides of the tensor.
 *
 * @return An NSArray of NSNumber objects representing the step sizes for each dimension.
 */
@property(nonatomic, readonly) NSArray<NSNumber *> *strides NS_REFINED_FOR_SWIFT;

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
@property(nonatomic, readonly) NSInteger count NS_REFINED_FOR_SWIFT;

/**
 * Initializes a tensor with a native TensorPtr instance.
 *
 * @param nativeInstance A pointer to a native TensorPtr instance.
 * @return An initialized ExecuTorchTensor instance.
 */
- (instancetype)initWithNativeInstance:(void *)nativeInstance
    NS_DESIGNATED_INITIALIZER
    NS_SWIFT_UNAVAILABLE("");

/**
 * Creates a new tensor that shares the underlying data storage with the
 * given tensor. This new tensor is a view and does not own the data.
 *
 * @param otherTensor The tensor instance to create a view of.
 * @return A new ExecuTorchTensor instance that shares data with otherTensor.
 */
- (instancetype)initWithTensor:(ExecuTorchTensor *)otherTensor
    NS_SWIFT_NAME(init(_:));

/**
 * Creates a deep copy of the tensor.
 * The new tensor will have its own copy of the data.
 *
 * @return A new ExecuTorchTensor instance that is a duplicate of the current tensor.
 */
- (instancetype)copy;

/**
 * Creates a deep copy of the tensor, potentially casting to a new data type.
 * The new tensor will have its own copy of the data.
 *
 * @param dataType The desired data type for the new tensor.
 * @return A new ExecuTorchTensor instance that is a duplicate (and possibly casted) of the current tensor.
*/
- (instancetype)copyToDataType:(ExecuTorchDataType)dataType
    NS_SWIFT_NAME(copy(to:));

/**
 * Executes a block with a pointer to the tensor's immutable byte data.
 *
 * @param handler A block that receives:
 *   - a pointer to the data,
 *   - the total number of elements,
 *   - and the data type.
 */
- (void)bytesWithHandler:(NS_NOESCAPE void (^)(const void *pointer, NSInteger count, ExecuTorchDataType dataType))handler
    NS_SWIFT_NAME(bytes(_:));

/**
 * Executes a block with a pointer to the tensor's mutable byte data.
 *
 * @param handler A block that receives:
 *   - a mutable pointer to the data,
 *   - the total number of elements,
 *   - and the data type.
 */
- (void)mutableBytesWithHandler:(NS_NOESCAPE void (^)(void *pointer, NSInteger count, ExecuTorchDataType dataType))handler
    NS_SWIFT_NAME(mutableBytes(_:));

/**
 * Resizes the tensor to a new shape.
 *
 * @param shape An NSArray of NSNumber objects representing the desired new shape.
 * @param error A pointer to an NSError pointer that is set if an error occurs.
 * @return YES if the tensor was successfully resized; otherwise, NO.
 */
- (BOOL)resizeToShape:(NSArray<NSNumber *> *)shape
                error:(NSError **)error
    NS_REFINED_FOR_SWIFT;

/**
 * Determines whether the current tensor is equal to another tensor.
 *
 * @param other Another ExecuTorchTensor instance to compare against.
 * @return YES if the tensors have the same data type, shape, dimension order,
 * strides, and underlying data; otherwise, NO.
 */
- (BOOL)isEqualToTensor:(nullable ExecuTorchTensor *)other
    NS_REFINED_FOR_SWIFT;

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
                      shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_REFINED_FOR_SWIFT;

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
                           dataType:(ExecuTorchDataType)dataType
    NS_SWIFT_UNAVAILABLE("");

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
                      shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_SWIFT_UNAVAILABLE("");

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
                           dataType:(ExecuTorchDataType)dataType
    NS_SWIFT_UNAVAILABLE("");

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
                shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_REFINED_FOR_SWIFT;

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
                     dataType:(ExecuTorchDataType)dataType
    NS_SWIFT_UNAVAILABLE("");

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
                shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_SWIFT_UNAVAILABLE("");

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
                     dataType:(ExecuTorchDataType)dataType
    NS_SWIFT_UNAVAILABLE("");

@end

#pragma mark - Data Category

@interface ExecuTorchTensor (Data)

/**
 * Initializes a tensor using an NSData object. The tensor will hold a
 * strong reference to the NSData object to manage the lifetime of the
 * underlying data buffer, which is not copied.
 *
 * @param data An NSData object containing the tensor data.
 * @param shape An NSArray of NSNumber objects representing the tensor's shape.
 * @param strides An NSArray of NSNumber objects representing the tensor's strides.
 * @param dimensionOrder An NSArray of NSNumber objects indicating the order of dimensions.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value indicating the shape dynamism.
 * @return An initialized ExecuTorchTensor instance using the provided data.
 */
- (instancetype)initWithData:(NSData *)data
                       shape:(NSArray<NSNumber *> *)shape
                     strides:(NSArray<NSNumber *> *)strides
              dimensionOrder:(NSArray<NSNumber *> *)dimensionOrder
                    dataType:(ExecuTorchDataType)dataType
               shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_REFINED_FOR_SWIFT;

/**
 * Initializes a tensor using an NSData object as the underlying data buffer with dynamic bound shape.
 *
 * @param data An NSData object containing the tensor data.
 * @param shape An NSArray of NSNumber objects representing the tensor's shape.
 * @param strides An NSArray of NSNumber objects representing the tensor's strides.
 * @param dimensionOrder An NSArray of NSNumber objects indicating the order of dimensions.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @return An initialized ExecuTorchTensor instance using the provided data.
 */
- (instancetype)initWithData:(NSData *)data
                       shape:(NSArray<NSNumber *> *)shape
                     strides:(NSArray<NSNumber *> *)strides
              dimensionOrder:(NSArray<NSNumber *> *)dimensionOrder
                    dataType:(ExecuTorchDataType)dataType
    NS_SWIFT_UNAVAILABLE("");

/**
 * Initializes a tensor using an NSData object as the underlying data buffer, specifying shape, data type, and explicit shape dynamism.
 *
 * @param data An NSData object containing the tensor data.
 * @param shape An NSArray of NSNumber objects representing the tensor's shape.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value indicating the shape dynamism.
 * @return An initialized ExecuTorchTensor instance using the provided data.
 */
- (instancetype)initWithData:(NSData *)data
                       shape:(NSArray<NSNumber *> *)shape
                    dataType:(ExecuTorchDataType)dataType
               shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_SWIFT_UNAVAILABLE("");

/**
 * Initializes a tensor using an NSData object as the underlying data buffer, specifying only the shape and data type.
 *
 * @param data An NSData object containing the tensor data.
 * @param shape An NSArray of NSNumber objects representing the tensor's shape.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @return An initialized ExecuTorchTensor instance using the provided data.
 */
- (instancetype)initWithData:(NSData *)data
                       shape:(NSArray<NSNumber *> *)shape
                    dataType:(ExecuTorchDataType)dataType
    NS_SWIFT_UNAVAILABLE("");

@end

#pragma mark - Scalars Category

@interface ExecuTorchTensor (Scalars)

/**
 * Initializes a tensor with an array of scalar values and full tensor properties.
 *
 * @param scalars An NSArray of NSNumber objects representing the scalar values.
 * @param shape An NSArray of NSNumber objects representing the desired tensor shape.
 * @param strides An NSArray of NSNumber objects representing the tensor strides.
 * @param dimensionOrder An NSArray of NSNumber objects indicating the order of dimensions.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value indicating the shape dynamism.
 * @return An initialized ExecuTorchTensor instance containing the provided scalar values.
 */
- (instancetype)initWithScalars:(NSArray<NSNumber *> *)scalars
                          shape:(NSArray<NSNumber *> *)shape
                        strides:(NSArray<NSNumber *> *)strides
                 dimensionOrder:(NSArray<NSNumber *> *)dimensionOrder
                       dataType:(ExecuTorchDataType)dataType
                  shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_SWIFT_UNAVAILABLE("");

/**
 * Initializes a tensor with an array of scalar values, specifying shape, strides, dimension order, and data type,
 * using a default dynamic bound shape for shape dynamism.
 *
 * @param scalars An NSArray of NSNumber objects representing the scalar values.
 * @param shape An NSArray of NSNumber objects representing the desired tensor shape.
 * @param strides An NSArray of NSNumber objects representing the tensor strides.
 * @param dimensionOrder An NSArray of NSNumber objects indicating the order of dimensions.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @return An initialized ExecuTorchTensor instance containing the scalar values.
 */
- (instancetype)initWithScalars:(NSArray<NSNumber *> *)scalars
                          shape:(NSArray<NSNumber *> *)shape
                        strides:(NSArray<NSNumber *> *)strides
                 dimensionOrder:(NSArray<NSNumber *> *)dimensionOrder
                       dataType:(ExecuTorchDataType)dataType
    NS_SWIFT_UNAVAILABLE("");

/**
 * Initializes a tensor with an array of scalar values, specifying the desired shape, data type, and explicit shape dynamism.
 *
 * @param scalars An NSArray of NSNumber objects representing the scalar values.
 * @param shape An NSArray of NSNumber objects representing the desired tensor shape.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value indicating the shape dynamism.
 * @return An initialized ExecuTorchTensor instance.
 */
- (instancetype)initWithScalars:(NSArray<NSNumber *> *)scalars
                          shape:(NSArray<NSNumber *> *)shape
                       dataType:(ExecuTorchDataType)dataType
                  shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_SWIFT_UNAVAILABLE("");

/**
 * Initializes a tensor with an array of scalar values and a specified shape,
 * using a default dynamic bound shape for shape dynamism.
 *
 * @param scalars An NSArray of NSNumber objects representing the scalar values.
 * @param shape An NSArray of NSNumber objects representing the desired tensor shape.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @return An initialized ExecuTorchTensor instance.
 */
- (instancetype)initWithScalars:(NSArray<NSNumber *> *)scalars
                          shape:(NSArray<NSNumber *> *)shape
                       dataType:(ExecuTorchDataType)dataType
    NS_SWIFT_UNAVAILABLE("");

/**
 * Initializes a tensor with an array of scalar values, specifying the tensor data type and explicit shape dynamism.
 * The shape is deduced from the count of the scalar array.
 *
 * @param scalars An NSArray of NSNumber objects representing the scalar values.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value indicating the shape dynamism.
 * @return An initialized ExecuTorchTensor instance with the shape deduced from the scalar count.
 */
- (instancetype)initWithScalars:(NSArray<NSNumber *> *)scalars
                       dataType:(ExecuTorchDataType)dataType
                  shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_SWIFT_UNAVAILABLE("");

/**
 * Initializes a tensor with an array of scalar values, specifying the tensor data type.
 * The shape is deduced from the count of the scalar array.
 *
 * @param scalars An NSArray of NSNumber objects representing the scalar values.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @return An initialized ExecuTorchTensor instance with the shape deduced from the scalar count.
 */
- (instancetype)initWithScalars:(NSArray<NSNumber *> *)scalars
                       dataType:(ExecuTorchDataType)dataType
    NS_SWIFT_UNAVAILABLE("");

/**
 * Initializes a tensor with an array of scalar values, a specified shape and explicit shape dynamism.
 * The data type is automatically deduced from the first element of the array.
 *
 * @param scalars An NSArray of NSNumber objects representing the scalar values.
 * @param shape An NSArray of NSNumber objects representing the desired tensor shape.
 * @param shapeDynamism An ExecuTorchShapeDynamism value indicating the shape dynamism.
 * @return An initialized ExecuTorchTensor instance.
 */
- (instancetype)initWithScalars:(NSArray<NSNumber *> *)scalars
                          shape:(NSArray<NSNumber *> *)shape
                  shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_SWIFT_UNAVAILABLE("");

/**
 * Initializes a tensor with an array of scalar values and a specified shape.
 * The data type is automatically deduced from the first element of the array.
 *
 * @param scalars An NSArray of NSNumber objects representing the scalar values.
 * @param shape An NSArray of NSNumber objects representing the desired tensor shape.
 * @return An initialized ExecuTorchTensor instance.
 */
- (instancetype)initWithScalars:(NSArray<NSNumber *> *)scalars
                          shape:(NSArray<NSNumber *> *)shape
    NS_SWIFT_UNAVAILABLE("");

/**
 * Initializes a tensor with an array of scalar values, automatically deducing the tensor shape and data type.
 *
 * @param scalars An NSArray of NSNumber objects representing the scalar values.
 * @return An initialized ExecuTorchTensor instance with shape and data type deduced.
 */
- (instancetype)initWithScalars:(NSArray<NSNumber *> *)scalars
    NS_SWIFT_UNAVAILABLE("");

@end

@interface ExecuTorchTensor (Scalar)

/**
 * Initializes a tensor with a single scalar value and a specified data type.
 *
 * @param scalar An NSNumber representing the scalar value.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @return An initialized ExecuTorchTensor instance representing the scalar.
 */
- (instancetype)initWithScalar:(NSNumber *)scalar
                      dataType:(ExecuTorchDataType)dataType
    NS_REFINED_FOR_SWIFT;

/**
 * Initializes a tensor with a single scalar value, automatically deducing its data type.
 *
 * @param scalar An NSNumber representing the scalar value.
 * @return An initialized ExecuTorchTensor instance representing the scalar.
 */
- (instancetype)initWithScalar:(NSNumber *)scalar
    NS_SWIFT_UNAVAILABLE("");

/** 
 * Initializes a tensor with a byte scalar value.
 *
 * @param scalar A uint8_t value.
 * @return An initialized ExecuTorchTensor instance.
 */
- (instancetype)initWithByte:(uint8_t)scalar
    NS_SWIFT_UNAVAILABLE("");

/** 
 * Initializes a tensor with a char scalar value.
 *
 * @param scalar An int8_t value.
 * @return An initialized ExecuTorchTensor instance.
 */
- (instancetype)initWithChar:(int8_t)scalar
    NS_SWIFT_UNAVAILABLE("");

/** 
 * Initializes a tensor with a short scalar value.
 *
 * @param scalar An int16_t value.
 * @return An initialized ExecuTorchTensor instance.
 */
- (instancetype)initWithShort:(int16_t)scalar
    NS_SWIFT_UNAVAILABLE("");

/** 
 * Initializes a tensor with an int scalar value.
 *
 * @param scalar An int32_t value.
 * @return An initialized ExecuTorchTensor instance.
 */
- (instancetype)initWithInt:(int32_t)scalar
    NS_SWIFT_UNAVAILABLE("");

/** 
 * Initializes a tensor with a long scalar value.
 *
 * @param scalar An int64_t value.
 * @return An initialized ExecuTorchTensor instance.
 */
- (instancetype)initWithLong:(int64_t)scalar
    NS_SWIFT_UNAVAILABLE("");

/** 
 * Initializes a tensor with a float scalar value.
 *
 * @param scalar A float value.
 * @return An initialized ExecuTorchTensor instance.
 */
- (instancetype)initWithFloat:(float)scalar
    NS_SWIFT_UNAVAILABLE("");

/** 
 * Initializes a tensor with a double scalar value.
 *
 * @param scalar A double value.
 * @return An initialized ExecuTorchTensor instance.
 */
- (instancetype)initWithDouble:(double)scalar
    NS_SWIFT_UNAVAILABLE("");

/** 
 * Initializes a tensor with a boolean scalar value.
 *
 * @param scalar A BOOL value.
 * @return An initialized ExecuTorchTensor instance.
 */
- (instancetype)initWithBool:(BOOL)scalar
    NS_SWIFT_UNAVAILABLE("");

/** 
 * Initializes a tensor with a uint16 scalar value.
 *
 * @param scalar A uint16_t value.
 * @return An initialized ExecuTorchTensor instance.
 */
- (instancetype)initWithUInt16:(uint16_t)scalar
    NS_SWIFT_NAME(init(_:));

/** 
 * Initializes a tensor with a uint32 scalar value.
 *
 * @param scalar A uint32_t value.
 * @return An initialized ExecuTorchTensor instance.
 */
- (instancetype)initWithUInt32:(uint32_t)scalar
    NS_SWIFT_NAME(init(_:));

/** 
 * Initializes a tensor with a uint64 scalar value.
 *
 * @param scalar A uint64_t value.
 * @return An initialized ExecuTorchTensor instance.
 */
- (instancetype)initWithUInt64:(uint64_t)scalar
    NS_SWIFT_NAME(init(_:));

/** 
 * Initializes a tensor with an NSInteger scalar value.
 *
 * @param scalar An NSInteger value.
 * @return An initialized ExecuTorchTensor instance.
 */
- (instancetype)initWithInteger:(NSInteger)scalar
    NS_SWIFT_NAME(init(_:));

/** 
 * Initializes a tensor with an NSUInteger scalar value.
 *
 * @param scalar An NSUInteger value.
 * @return An initialized ExecuTorchTensor instance.
 */
- (instancetype)initWithUnsignedInteger:(NSUInteger)scalar
    NS_SWIFT_NAME(init(_:));

@end

#pragma mark - Empty Category

@interface ExecuTorchTensor (Empty)

/**
 * Creates an empty tensor with the specified shape, strides, data type, and shape dynamism.
 *
 * @param shape An NSArray of NSNumber objects representing the desired shape.
 * @param strides An NSArray of NSNumber objects representing the desired strides.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value specifying whether the shape is static or dynamic.
 * @return A new, empty ExecuTorchTensor instance.
 */
+ (instancetype)emptyTensorWithShape:(NSArray<NSNumber *> *)shape
                             strides:(NSArray<NSNumber *> *)strides
                            dataType:(ExecuTorchDataType)dataType
                       shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_REFINED_FOR_SWIFT
    NS_RETURNS_RETAINED;

/**
 * Creates an empty tensor with the specified shape, data type, and shape dynamism.
 *
 * @param shape An NSArray of NSNumber objects representing the desired shape.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value specifying whether the shape is static or dynamic.
 * @return A new, empty ExecuTorchTensor instance.
 */
+ (instancetype)emptyTensorWithShape:(NSArray<NSNumber *> *)shape
                            dataType:(ExecuTorchDataType)dataType
                       shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Creates an empty tensor with the specified shape and data type, using dynamic bound shape.
 *
 * @param shape An NSArray of NSNumber objects representing the desired shape.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @return A new, empty ExecuTorchTensor instance.
 */
+ (instancetype)emptyTensorWithShape:(NSArray<NSNumber *> *)shape
                            dataType:(ExecuTorchDataType)dataType
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Creates an empty tensor similar to the given tensor, with the specified data type and shape dynamism.
 *
 * @param tensor An existing ExecuTorchTensor instance whose shape and strides are used.
 * @param dataType An ExecuTorchDataType value specifying the desired element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value specifying whether the shape is static or dynamic.
 * @return A new, empty ExecuTorchTensor instance with the same shape as the provided tensor.
 */
+ (instancetype)emptyTensorLikeTensor:(ExecuTorchTensor *)tensor
                             dataType:(ExecuTorchDataType)dataType
                        shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_REFINED_FOR_SWIFT
    NS_RETURNS_RETAINED;

/**
 * Creates an empty tensor similar to the given tensor, with the specified data type.
 *
 * @param tensor An existing ExecuTorchTensor instance whose shape and strides are used.
 * @param dataType An ExecuTorchDataType value specifying the desired element type.
 * @return A new, empty ExecuTorchTensor instance with the same shape as the provided tensor.
 */
+ (instancetype)emptyTensorLikeTensor:(ExecuTorchTensor *)tensor
                             dataType:(ExecuTorchDataType)dataType
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Creates an empty tensor similar to the given tensor.
 *
 * @param tensor An existing ExecuTorchTensor instance.
 * @return A new, empty ExecuTorchTensor instance with the same properties as the provided tensor.
 */
+ (instancetype)emptyTensorLikeTensor:(ExecuTorchTensor *)tensor
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

@end

#pragma mark - Full Category

@interface ExecuTorchTensor (Full)

/**
 * Creates a tensor filled with the specified scalar value, with full specification of shape, strides, data type, and shape dynamism.
 *
 * @param shape An NSArray of NSNumber objects representing the desired shape.
 * @param scalar An NSNumber representing the value to fill the tensor.
 * @param strides An NSArray of NSNumber objects representing the desired strides.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value specifying whether the shape is static or dynamic.
 * @return A new ExecuTorchTensor instance filled with the scalar value.
 */
+ (instancetype)fullTensorWithShape:(NSArray<NSNumber *> *)shape
                             scalar:(NSNumber *)scalar
                            strides:(NSArray<NSNumber *> *)strides
                           dataType:(ExecuTorchDataType)dataType
                      shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_REFINED_FOR_SWIFT
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor filled with the specified scalar value, with the given shape, data type, and shape dynamism.
 *
 * @param shape An NSArray of NSNumber objects representing the desired shape.
 * @param scalar An NSNumber representing the value to fill the tensor.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value specifying whether the shape is static or dynamic.
 * @return A new ExecuTorchTensor instance filled with the scalar value.
 */
+ (instancetype)fullTensorWithShape:(NSArray<NSNumber *> *)shape
                             scalar:(NSNumber *)scalar
                           dataType:(ExecuTorchDataType)dataType
                      shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor filled with the specified scalar value, with the given shape and data type,
 * using dynamic bound shape for strides and dimension order.
 *
 * @param shape An NSArray of NSNumber objects representing the desired shape.
 * @param scalar An NSNumber representing the value to fill the tensor.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @return A new ExecuTorchTensor instance filled with the scalar value.
 */
+ (instancetype)fullTensorWithShape:(NSArray<NSNumber *> *)shape
                             scalar:(NSNumber *)scalar
                           dataType:(ExecuTorchDataType)dataType
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor filled with the specified scalar value, similar to an existing tensor, with the given data type and shape dynamism.
 *
 * @param tensr An existing ExecuTorchTensor instance whose shape and strides are used.
 * @param scalar An NSNumber representing the value to fill the tensor.
 * @param dataType An ExecuTorchDataType value specifying the desired element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value specifying whether the shape is static or dynamic.
 * @return A new ExecuTorchTensor instance filled with the scalar value.
 */
+ (instancetype)fullTensorLikeTensor:(ExecuTorchTensor *)tensr
                              scalar:(NSNumber *)scalar
                            dataType:(ExecuTorchDataType)dataType
                       shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_REFINED_FOR_SWIFT
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor filled with the specified scalar value, similar to an existing tensor, with the given data type.
 *
 * @param tensr An existing ExecuTorchTensor instance whose shape and strides are used.
 * @param scalar An NSNumber representing the value to fill the tensor.
 * @param dataType An ExecuTorchDataType value specifying the desired element type.
 * @return A new ExecuTorchTensor instance filled with the scalar value.
 */
+ (instancetype)fullTensorLikeTensor:(ExecuTorchTensor *)tensr
                              scalar:(NSNumber *)scalar
                            dataType:(ExecuTorchDataType)dataType
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor filled with the specified scalar value, similar to an existing tensor.
 *
 * @param tensr An existing ExecuTorchTensor instance.
 * @param scalar An NSNumber representing the value to fill the tensor.
 * @return A new ExecuTorchTensor instance filled with the scalar value.
 */
+ (instancetype)fullTensorLikeTensor:(ExecuTorchTensor *)tensr
                              scalar:(NSNumber *)scalar
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

@end

#pragma mark - Ones Category

@interface ExecuTorchTensor (Ones)

/**
 * Creates a tensor filled with ones, with the specified shape, data type, and shape dynamism.
 *
 * @param shape An NSArray of NSNumber objects representing the desired shape.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value specifying whether the shape is static or dynamic.
 * @return A new ExecuTorchTensor instance filled with ones.
 */
+ (instancetype)onesTensorWithShape:(NSArray<NSNumber *> *)shape
                           dataType:(ExecuTorchDataType)dataType
                      shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_REFINED_FOR_SWIFT
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor filled with ones, with the specified shape and data type.
 *
 * @param shape An NSArray of NSNumber objects representing the desired shape.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @return A new ExecuTorchTensor instance filled with ones.
 */
+ (instancetype)onesTensorWithShape:(NSArray<NSNumber *> *)shape
                           dataType:(ExecuTorchDataType)dataType
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor filled with ones similar to an existing tensor, with the specified data type and shape dynamism.
 *
 * @param tensor An existing ExecuTorchTensor instance whose shape and strides are used.
 * @param dataType An ExecuTorchDataType value specifying the desired element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value specifying whether the shape is static or dynamic.
 * @return A new ExecuTorchTensor instance filled with ones.
 */
+ (instancetype)onesTensorLikeTensor:(ExecuTorchTensor *)tensor
                            dataType:(ExecuTorchDataType)dataType
                       shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_REFINED_FOR_SWIFT
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor filled with ones similar to an existing tensor, with the specified data type.
 *
 * @param tensor An existing ExecuTorchTensor instance whose shape and strides are used.
 * @param dataType An ExecuTorchDataType value specifying the desired element type.
 * @return A new ExecuTorchTensor instance filled with ones.
 */
+ (instancetype)onesTensorLikeTensor:(ExecuTorchTensor *)tensor
                            dataType:(ExecuTorchDataType)dataType
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor filled with ones similar to an existing tensor.
 *
 * @param tensor An existing ExecuTorchTensor instance.
 * @return A new ExecuTorchTensor instance filled with ones.
 */
+ (instancetype)onesTensorLikeTensor:(ExecuTorchTensor *)tensor
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

@end

#pragma mark - Zeros Category

@interface ExecuTorchTensor (Zeros)

/**
 * Creates a tensor filled with zeros, with the specified shape, data type, and shape dynamism.
 *
 * @param shape An NSArray of NSNumber objects representing the desired shape.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value specifying whether the shape is static or dynamic.
 * @return A new ExecuTorchTensor instance filled with zeros.
 */
+ (instancetype)zerosTensorWithShape:(NSArray<NSNumber *> *)shape
                            dataType:(ExecuTorchDataType)dataType
                       shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_REFINED_FOR_SWIFT
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor filled with zeros, with the specified shape and data type.
 *
 * @param shape An NSArray of NSNumber objects representing the desired shape.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @return A new ExecuTorchTensor instance filled with zeros.
 */
+ (instancetype)zerosTensorWithShape:(NSArray<NSNumber *> *)shape
                            dataType:(ExecuTorchDataType)dataType
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor filled with zeros similar to an existing tensor, with the specified data type and shape dynamism.
 *
 * @param tensor An existing ExecuTorchTensor instance whose shape and strides are used.
 * @param dataType An ExecuTorchDataType value specifying the desired element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value specifying whether the shape is static or dynamic.
 * @return A new ExecuTorchTensor instance filled with zeros.
 */
+ (instancetype)zerosTensorLikeTensor:(ExecuTorchTensor *)tensor
                             dataType:(ExecuTorchDataType)dataType
                        shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_REFINED_FOR_SWIFT
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor filled with zeros similar to an existing tensor, with the specified data type.
 *
 * @param tensor An existing ExecuTorchTensor instance whose shape and strides are used.
 * @param dataType An ExecuTorchDataType value specifying the desired element type.
 * @return A new ExecuTorchTensor instance filled with zeros.
 */
+ (instancetype)zerosTensorLikeTensor:(ExecuTorchTensor *)tensor
                             dataType:(ExecuTorchDataType)dataType
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor filled with zeros similar to an existing tensor.
 *
 * @param tensor An existing ExecuTorchTensor instance.
 * @return A new ExecuTorchTensor instance filled with zeros.
 */
+ (instancetype)zerosTensorLikeTensor:(ExecuTorchTensor *)tensor
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

@end

#pragma mark - Random Category

@interface ExecuTorchTensor (Random)

/**
 * Creates a tensor with random values, with full specification of shape, strides, data type, and shape dynamism.
 *
 * @param shape An NSArray of NSNumber objects representing the desired shape.
 * @param strides An NSArray of NSNumber objects representing the desired strides.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value specifying whether the shape is static or dynamic.
 * @return A new ExecuTorchTensor instance filled with random values.
 */
+ (instancetype)randomTensorWithShape:(NSArray<NSNumber *> *)shape
                              strides:(NSArray<NSNumber *> *)strides
                             dataType:(ExecuTorchDataType)dataType
                        shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_REFINED_FOR_SWIFT
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor with random values, with the specified shape and data type.
 *
 * @param shape An NSArray of NSNumber objects representing the desired shape.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value specifying whether the shape is static or dynamic.
 * @return A new ExecuTorchTensor instance filled with random values.
 */
+ (instancetype)randomTensorWithShape:(NSArray<NSNumber *> *)shape
                             dataType:(ExecuTorchDataType)dataType
                        shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor with random values, with the specified shape (using dynamic bound shape) and data type.
 *
 * @param shape An NSArray of NSNumber objects representing the desired shape.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @return A new ExecuTorchTensor instance filled with random values.
 */
+ (instancetype)randomTensorWithShape:(NSArray<NSNumber *> *)shape
                             dataType:(ExecuTorchDataType)dataType
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor with random values similar to an existing tensor, with the specified data type and shape dynamism.
 *
 * @param tensor An existing ExecuTorchTensor instance whose shape and strides are used.
 * @param dataType An ExecuTorchDataType value specifying the desired element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value specifying whether the shape is static or dynamic.
 * @return A new ExecuTorchTensor instance filled with random values.
 */
+ (instancetype)randomTensorLikeTensor:(ExecuTorchTensor *)tensor
                              dataType:(ExecuTorchDataType)dataType
                         shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_REFINED_FOR_SWIFT
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor with random values similar to an existing tensor, with the specified data type.
 *
 * @param tensor An existing ExecuTorchTensor instance whose shape and strides are used.
 * @param dataType An ExecuTorchDataType value specifying the desired element type.
 * @return A new ExecuTorchTensor instance filled with random values.
 */
+ (instancetype)randomTensorLikeTensor:(ExecuTorchTensor *)tensor
                              dataType:(ExecuTorchDataType)dataType
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor with random values similar to an existing tensor.
 *
 * @param tensor An existing ExecuTorchTensor instance.
 * @return A new ExecuTorchTensor instance filled with random values.
 */
+ (instancetype)randomTensorLikeTensor:(ExecuTorchTensor *)tensor
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

@end

#pragma mark - RandomNormal Category

@interface ExecuTorchTensor (RandomNormal)

/**
 * Creates a tensor with random values drawn from a normal distribution,
 * with full specification of shape, strides, data type, and shape dynamism.
 *
 * @param shape An NSArray of NSNumber objects representing the desired shape.
 * @param strides An NSArray of NSNumber objects representing the desired strides.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value specifying whether the shape is static or dynamic.
 * @return A new ExecuTorchTensor instance filled with values from a normal distribution.
 */
+ (instancetype)randomNormalTensorWithShape:(NSArray<NSNumber *> *)shape
                                    strides:(NSArray<NSNumber *> *)strides
                                   dataType:(ExecuTorchDataType)dataType
                              shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_REFINED_FOR_SWIFT
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor with random values drawn from a normal distribution,
 * with the specified shape and data type.
 *
 * @param shape An NSArray of NSNumber objects representing the desired shape.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value specifying whether the shape is static or dynamic.
 * @return A new ExecuTorchTensor instance filled with values from a normal distribution.
 */
+ (instancetype)randomNormalTensorWithShape:(NSArray<NSNumber *> *)shape
                                   dataType:(ExecuTorchDataType)dataType
                              shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor with random values drawn from a normal distribution,
 * with the specified shape (using dynamic bound shape) and data type.
 *
 * @param shape An NSArray of NSNumber objects representing the desired shape.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @return A new ExecuTorchTensor instance filled with values from a normal distribution.
 */
+ (instancetype)randomNormalTensorWithShape:(NSArray<NSNumber *> *)shape
                                   dataType:(ExecuTorchDataType)dataType
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor with random normal values similar to an existing tensor,
 * with the specified data type and shape dynamism.
 *
 * @param tensor An existing ExecuTorchTensor instance whose shape and strides are used.
 * @param dataType An ExecuTorchDataType value specifying the desired element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value specifying whether the shape is static or dynamic.
 * @return A new ExecuTorchTensor instance filled with values from a normal distribution.
 */
+ (instancetype)randomNormalTensorLikeTensor:(ExecuTorchTensor *)tensor
                                    dataType:(ExecuTorchDataType)dataType
                               shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_REFINED_FOR_SWIFT
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor with random normal values similar to an existing tensor,
 * with the specified data type.
 *
 * @param tensor An existing ExecuTorchTensor instance whose shape and strides are used.
 * @param dataType An ExecuTorchDataType value specifying the desired element type.
 * @return A new ExecuTorchTensor instance filled with values from a normal distribution.
 */
+ (instancetype)randomNormalTensorLikeTensor:(ExecuTorchTensor *)tensor
                                    dataType:(ExecuTorchDataType)dataType
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor with random normal values similar to an existing tensor.
 *
 * @param tensor An existing ExecuTorchTensor instance.
 * @return A new ExecuTorchTensor instance filled with values from a normal distribution.
 */
+ (instancetype)randomNormalTensorLikeTensor:(ExecuTorchTensor *)tensor
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

@end

#pragma mark - RandomInteger Category

@interface ExecuTorchTensor (RandomInteger)

/**
 * Creates a tensor with random integer values in the specified range,
 * with full specification of shape, strides, data type, and shape dynamism.
 *
 * @param low An NSInteger specifying the inclusive lower bound of random values.
 * @param high An NSInteger specifying the exclusive upper bound of random values.
 * @param shape An NSArray of NSNumber objects representing the desired shape.
 * @param strides An NSArray of NSNumber objects representing the desired strides.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value specifying whether the shape is static or dynamic.
 * @return A new ExecuTorchTensor instance filled with random integer values.
 */
+ (instancetype)randomIntegerTensorWithLow:(NSInteger)low
                                      high:(NSInteger)high
                                     shape:(NSArray<NSNumber *> *)shape
                                   strides:(NSArray<NSNumber *> *)strides
                                  dataType:(ExecuTorchDataType)dataType
                             shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_REFINED_FOR_SWIFT
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor with random integer values in the specified range,
 * with the given shape and data type.
 *
 * @param low An NSInteger specifying the inclusive lower bound of random values.
 * @param high An NSInteger specifying the exclusive upper bound of random values.
 * @param shape An NSArray of NSNumber objects representing the desired shape.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value specifying whether the shape is static or dynamic.
 * @return A new ExecuTorchTensor instance filled with random integer values.
 */
+ (instancetype)randomIntegerTensorWithLow:(NSInteger)low
                                      high:(NSInteger)high
                                     shape:(NSArray<NSNumber *> *)shape
                                  dataType:(ExecuTorchDataType)dataType
                             shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor with random integer values in the specified range,
 * with the given shape (using dynamic bound shape) and data type.
 *
 * @param low An NSInteger specifying the inclusive lower bound of random values.
 * @param high An NSInteger specifying the exclusive upper bound of random values.
 * @param shape An NSArray of NSNumber objects representing the desired shape.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @return A new ExecuTorchTensor instance filled with random integer values.
 */
+ (instancetype)randomIntegerTensorWithLow:(NSInteger)low
                                      high:(NSInteger)high
                                     shape:(NSArray<NSNumber *> *)shape
                                  dataType:(ExecuTorchDataType)dataType
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor with random integer values in the specified range, similar to an existing tensor,
 * with the given data type and shape dynamism.
 *
 * @param tensor An existing ExecuTorchTensor instance whose shape and strides are used.
 * @param low An NSInteger specifying the inclusive lower bound of random values.
 * @param high An NSInteger specifying the exclusive upper bound of random values.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @param shapeDynamism An ExecuTorchShapeDynamism value specifying whether the shape is static or dynamic.
 * @return A new ExecuTorchTensor instance filled with random integer values.
 */
+ (instancetype)randomIntegerTensorLikeTensor:(ExecuTorchTensor *)tensor
                                          low:(NSInteger)low
                                         high:(NSInteger)high
                                     dataType:(ExecuTorchDataType)dataType
                                shapeDynamism:(ExecuTorchShapeDynamism)shapeDynamism
    NS_REFINED_FOR_SWIFT
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor with random integer values in the specified range, similar to an existing tensor,
 * with the given data type.
 *
 * @param tensor An existing ExecuTorchTensor instance whose shape and strides are used.
 * @param low An NSInteger specifying the inclusive lower bound of random values.
 * @param high An NSInteger specifying the exclusive upper bound of random values.
 * @param dataType An ExecuTorchDataType value specifying the element type.
 * @return A new ExecuTorchTensor instance filled with random integer values.
 */
+ (instancetype)randomIntegerTensorLikeTensor:(ExecuTorchTensor *)tensor
                                          low:(NSInteger)low
                                         high:(NSInteger)high
                                     dataType:(ExecuTorchDataType)dataType
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

/**
 * Creates a tensor with random integer values in the specified range, similar to an existing tensor.
 *
 * @param tensor An existing ExecuTorchTensor instance.
 * @param low An NSInteger specifying the inclusive lower bound of random values.
 * @param high An NSInteger specifying the exclusive upper bound of random values.
 * @return A new ExecuTorchTensor instance filled with random integer values.
 */
+ (instancetype)randomIntegerTensorLikeTensor:(ExecuTorchTensor *)tensor
                                          low:(NSInteger)low
                                         high:(NSInteger)high
    NS_SWIFT_UNAVAILABLE("")
    NS_RETURNS_RETAINED;

@end

NS_ASSUME_NONNULL_END
