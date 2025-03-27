/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchValue.h"

#import <executorch/runtime/platform/assert.h>

@interface ExecuTorchValue ()

- (instancetype)initWithTag:(ExecuTorchValueTag)tag
                      value:(nullable id)value NS_DESIGNATED_INITIALIZER;

@end

@implementation ExecuTorchValue {
  ExecuTorchValueTag _tag;
  id _value;
}

+ (instancetype)valueWithTensor:(ExecuTorchTensor *)value {
  ET_CHECK(value);
  return [[ExecuTorchValue alloc] initWithTag:ExecuTorchValueTagTensor value:value];
}

+ (instancetype)valueWithString:(ExecuTorchStringValue)value {
  ET_CHECK(value);
  return [[ExecuTorchValue alloc] initWithTag:ExecuTorchValueTagString value:value];
}

+ (instancetype)valueWithBoolean:(ExecuTorchBooleanValue)value {
  return [[ExecuTorchValue alloc] initWithTag:ExecuTorchValueTagBoolean value:@(value)];
}

+ (instancetype)valueWithInteger:(ExecuTorchIntegerValue)value {
  return [[ExecuTorchValue alloc] initWithTag:ExecuTorchValueTagInteger value:@(value)];
}

+ (instancetype)valueWithDouble:(ExecuTorchDoubleValue)value {
  return [[ExecuTorchValue alloc] initWithTag:ExecuTorchValueTagDouble value:@(value)];
}

- (instancetype)init {
  return [self initWithTag:ExecuTorchValueTagNone value:nil];
}

- (instancetype)initWithTag:(ExecuTorchValueTag)tag
                      value:(nullable id)value {
  if (self = [super init]) {
    _tag = tag;
    _value = value;
  }
  return self;
}

- (ExecuTorchValueTag)tag {
  return _tag;
}

- (nullable ExecuTorchTensor *)tensorValue {
  return self.isTensor ? _value : nil;
}

- (nullable ExecuTorchScalarValue)scalarValue {
  return self.isScalar ? _value : nil;
}

- (nullable ExecuTorchStringValue)stringValue {
    return self.isString ? _value : nil;
}

- (ExecuTorchBooleanValue)boolValue {
  ET_CHECK(self.isBoolean);
  return [(ExecuTorchScalarValue)_value boolValue];
}

- (ExecuTorchIntegerValue)intValue {
  ET_CHECK(self.isInteger);
  return [(ExecuTorchScalarValue)_value integerValue];
}

- (ExecuTorchDoubleValue)doubleValue {
  ET_CHECK(self.isDouble);
  return [(ExecuTorchScalarValue)_value doubleValue];
}

- (BOOL)isNone {
  return _tag == ExecuTorchValueTagNone;
}

- (BOOL)isTensor {
  return _tag == ExecuTorchValueTagTensor;
}

- (BOOL)isScalar {
  return _tag == ExecuTorchValueTagBoolean ||
         _tag == ExecuTorchValueTagInteger ||
         _tag == ExecuTorchValueTagDouble;
}

- (BOOL)isString {
  return _tag == ExecuTorchValueTagString;
}

- (BOOL)isBoolean {
  return _tag == ExecuTorchValueTagBoolean;
}

- (BOOL)isInteger {
  return _tag == ExecuTorchValueTagInteger;
}

- (BOOL)isDouble {
  return _tag == ExecuTorchValueTagDouble;
}

- (BOOL)isEqualToValue:(nullable ExecuTorchValue *)other {
  if (!other) {
    return NO;
  }
  if (_tag != other->_tag) {
    return NO;
  }
  if (_value == nil) {
    return other->_value == nil;
  }
  return [_value isEqual:other->_value];
}

- (BOOL)isEqual:(nullable id)other {
  if (self == other) {
    return YES;
  }
  if (![other isKindOfClass:[ExecuTorchValue class]]) {
    return NO;
  }
  return [self isEqualToValue:(ExecuTorchValue *)other];
}

@end
