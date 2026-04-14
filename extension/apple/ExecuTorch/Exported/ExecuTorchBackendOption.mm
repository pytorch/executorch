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

@end
