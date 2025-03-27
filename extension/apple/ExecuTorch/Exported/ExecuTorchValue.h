/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchTensor.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Enum to define the dynamic type of a Value.
 * Values can be a subset, but must numerically match exactly those defined in
 * runtime/core/tag.h
 */
typedef NS_ENUM(uint32_t, ExecuTorchValueTag) {
  ExecuTorchValueTagNone,
  ExecuTorchValueTagTensor,
  ExecuTorchValueTagString,
  ExecuTorchValueTagDouble,
  ExecuTorchValueTagInteger,
  ExecuTorchValueTagBoolean,
  ExecuTorchValueTagBooleanList,
  ExecuTorchValueTagDoubleList,
  ExecuTorchValueTagIntegerList,
  ExecuTorchValueTagTensorList,
  ExecuTorchValueTagScalarList,
  ExecuTorchValueTagOptionalTensorList,
} NS_SWIFT_NAME(ValueTag);

typedef NSNumber *ExecuTorchScalarValue
    NS_SWIFT_BRIDGED_TYPEDEF NS_SWIFT_NAME(ScalarValue);
typedef NSString *ExecuTorchStringValue
    NS_SWIFT_BRIDGED_TYPEDEF NS_SWIFT_NAME(StringValue);
typedef BOOL ExecuTorchBooleanValue
    NS_SWIFT_BRIDGED_TYPEDEF NS_SWIFT_NAME(BoolValue);
typedef NSInteger ExecuTorchIntegerValue
    NS_SWIFT_BRIDGED_TYPEDEF NS_SWIFT_NAME(IntegerValue);
typedef double ExecuTorchDoubleValue
    NS_SWIFT_BRIDGED_TYPEDEF NS_SWIFT_NAME(DoubleValue);

/**
 * A dynamic value type used by ExecuTorch.
 *
 * ExecuTorchValue encapsulates a value that may be of various types such as
 * a tensor or a scalar. The value’s type is indicated by its tag.
 */
NS_SWIFT_NAME(Value)
__attribute__((deprecated("This API is experimental.")))
@interface ExecuTorchValue : NSObject

/**
 * The tag that indicates the dynamic type of the value.
 *
 * @return An ExecuTorchValueTag value.
 */
@property(nonatomic, readonly) ExecuTorchValueTag tag;

/**
 * The tensor value if the tag is ExecuTorchValueTagTensor.
 *
 * @return A Tensor instance or nil.
 */
@property(nullable, nonatomic, readonly) ExecuTorchTensor *tensorValue NS_SWIFT_NAME(tensor);

/**
 * The string value if the tag is ExecuTorchValueTagString.
 *
 * @return An NSString instance or nil.
 */
@property(nullable, nonatomic, readonly) ExecuTorchStringValue stringValue NS_SWIFT_NAME(string);

/**
 * The scalar value if the tag is boolean, integer or double.
 *
 * @return A scalar value or nil.
 */
@property(nullable, nonatomic, readonly) ExecuTorchScalarValue scalarValue NS_SWIFT_NAME(scalar);

/**
 * The boolean value if the tag is ExecuTorchValueTagBoolean.
 *
 * @return A BOOL representing the boolean value.
 */
@property(nonatomic, readonly) ExecuTorchBooleanValue boolValue NS_SWIFT_NAME(boolean);

/**
 * The integer value if the tag is ExecuTorchValueTagInteger.
 *
 * @return An NSInteger representing the integer value.
 */
@property(nonatomic, readonly) ExecuTorchIntegerValue intValue NS_SWIFT_NAME(integer);

/**
 * The double value if the tag is ExecuTorchValueTagDouble.
 *
 * @return A double representing the double value.
 */
@property(nonatomic, readonly) ExecuTorchDoubleValue doubleValue NS_SWIFT_NAME(double);

/**
 * Returns YES if the value is of type None.
 *
 * @return A BOOL indicating whether the value is None.
 */
@property(nonatomic, readonly) BOOL isNone;

/**
 * Returns YES if the value is a Tensor.
 *
 * @return A BOOL indicating whether the value is a Tensor.
 */
@property(nonatomic, readonly) BOOL isTensor;

/**
 * Returns YES if the value is a string.
 *
 * @return A BOOL indicating whether the value is a string.
 */
@property(nonatomic, readonly) BOOL isString;

/**
 * Returns YES if the value is a scalar (boolean, integer or double).
 *
 * @return A BOOL indicating whether the value is a scalar.
 */
@property(nonatomic, readonly) BOOL isScalar;

/**
 * Returns YES if the value is a boolean.
 *
 * @return A BOOL indicating whether the value is a boolean.
 */
@property(nonatomic, readonly) BOOL isBoolean;

/**
 * Returns YES if the value is an integer.
 *
 * @return A BOOL indicating whether the value is an integer.
 */
@property(nonatomic, readonly) BOOL isInteger;

/**
 * Returns YES if the value is a double.
 *
 * @return A BOOL indicating whether the value is a double.
 */
@property(nonatomic, readonly) BOOL isDouble;

/**
 * Creates an instance encapsulating a Tensor.
 *
 * @param value An ExecuTorchTensor instance.
 * @return A new ExecuTorchValue instance with a tag of ExecuTorchValueTagTensor.
 */
+ (instancetype)valueWithTensor:(ExecuTorchTensor *)value NS_SWIFT_NAME(init(_:));

/**
 * Creates an instance encapsulating a string.
 *
 * @param value A string.
 * @return A new ExecuTorchValue instance with a tag of ExecuTorchValueTagString.
 */
+ (instancetype)valueWithString:(ExecuTorchStringValue)value
    NS_SWIFT_NAME(init(_:));

/**
 * Creates an instance encapsulating a boolean.
 *
 * @param value A boolean.
 * @return A new ExecuTorchValue instance with a tag of ExecuTorchValueTagBoolean.
 */
+ (instancetype)valueWithBoolean:(ExecuTorchBooleanValue)value
    NS_SWIFT_NAME(init(_:));

/**
 * Creates an instance encapsulating an integer.
 *
 * @param value An integer.
 * @return A new ExecuTorchValue instance with a tag of ExecuTorchValueTagInteger.
 */
+ (instancetype)valueWithInteger:(ExecuTorchIntegerValue)value
    NS_SWIFT_NAME(init(_:));

/**
 * Creates an instance encapsulating a double value.
 *
 * @param value A double value.
 * @return A new ExecuTorchValue instance with a tag of ExecuTorchValueTagDouble.
 */
+ (instancetype)valueWithDouble:(ExecuTorchDoubleValue)value
    NS_SWIFT_NAME(init(_:));


/**
 * Determines whether the current Value is equal to another Value.
 *
 * @param other Another ExecuTorchValue instance to compare against.
 * @return YES if the values have the same tag and equal underlying values; otherwise, NO.
 */
- (BOOL)isEqualToValue:(nullable ExecuTorchValue *)other;

@end

NS_ASSUME_NONNULL_END
