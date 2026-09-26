/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchBackendOption.h"

@implementation ExecuTorchBackendOption {
  NSString *_key;
  ExecuTorchBackendOptionType _type;
  BOOL _boolValue;
  NSInteger _intValue;
  NSString *_stringValue;
}

- (instancetype)initWithKey:(NSString *)key
               booleanValue:(BOOL)value {
  self = [super init];
  if (self) {
    _key = [key copy];
    _type = ExecuTorchBackendOptionTypeBoolean;
    _boolValue = value;
  }
  return self;
}

- (instancetype)initWithKey:(NSString *)key
               integerValue:(NSInteger)value {
  self = [super init];
  if (self) {
    _key = [key copy];
    _type = ExecuTorchBackendOptionTypeInteger;
    _intValue = value;
  }
  return self;
}

- (instancetype)initWithKey:(NSString *)key
                stringValue:(NSString *)value {
  self = [super init];
  if (self) {
    _key = [key copy];
    _type = ExecuTorchBackendOptionTypeString;
    _stringValue = [value copy];
  }
  return self;
}

+ (instancetype)optionWithKey:(NSString *)key
                 booleanValue:(BOOL)value {
  return [[self alloc] initWithKey:key booleanValue:value];
}

+ (instancetype)optionWithKey:(NSString *)key
                 integerValue:(NSInteger)value {
  return [[self alloc] initWithKey:key integerValue:value];
}

+ (instancetype)optionWithKey:(NSString *)key
                  stringValue:(NSString *)value {
  return [[self alloc] initWithKey:key stringValue:value];
}

#pragma mark - NSObject

- (NSString *)description {
  switch (_type) {
    case ExecuTorchBackendOptionTypeBoolean:
      return [NSString stringWithFormat:@"<%@ %@=%@ (bool)>",
              NSStringFromClass([self class]), _key, _boolValue ? @"true" : @"false"];
    case ExecuTorchBackendOptionTypeInteger:
      return [NSString stringWithFormat:@"<%@ %@=%ld (int)>",
              NSStringFromClass([self class]), _key, (long)_intValue];
    case ExecuTorchBackendOptionTypeString:
      return [NSString stringWithFormat:@"<%@ %@=%@ (string)>",
              NSStringFromClass([self class]), _key,
              _stringValue ? [NSString stringWithFormat:@"\"%@\"", _stringValue] : @"(null)"];
  }
  return [super description];
}

- (NSString *)debugDescription {
  return [self description];
}

- (BOOL)isEqual:(id)object {
  if (self == object) {
    return YES;
  }
  if (![object isKindOfClass:[ExecuTorchBackendOption class]]) {
    return NO;
  }
  ExecuTorchBackendOption *other = (ExecuTorchBackendOption *)object;
  if (_type != other.type || ![_key isEqualToString:other.key]) {
    return NO;
  }
  switch (_type) {
    case ExecuTorchBackendOptionTypeBoolean:
      return _boolValue == other.boolValue;
    case ExecuTorchBackendOptionTypeInteger:
      return _intValue == other.intValue;
    case ExecuTorchBackendOptionTypeString: {
      // Both are non-null when type is String (init enforces it), but be
      // defensive in case of subclass/manual misuse.
      NSString *otherString = other.stringValue;
      if (_stringValue == otherString) {
        return YES;
      }
      if (_stringValue == nil || otherString == nil) {
        return NO;
      }
      return [_stringValue isEqualToString:otherString];
    }
  }
  return NO;
}

- (NSUInteger)hash {
  NSUInteger h = _key.hash ^ (NSUInteger)_type;
  switch (_type) {
    case ExecuTorchBackendOptionTypeBoolean:
      h ^= (NSUInteger)(_boolValue ? 1 : 0);
      break;
    case ExecuTorchBackendOptionTypeInteger:
      h ^= (NSUInteger)_intValue;
      break;
    case ExecuTorchBackendOptionTypeString:
      h ^= _stringValue.hash;
      break;
  }
  return h;
}

@end
