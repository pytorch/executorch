/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchValue.h"

@interface ExecuTorchValue ()

- (instancetype)initWithTag:(ExecuTorchValueTag)tag
                      value:(nullable id)value NS_DESIGNATED_INITIALIZER;

@end

@implementation ExecuTorchValue {
  ExecuTorchValueTag _tag;
  id _value;
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

- (BOOL)isNone {
  return _tag == ExecuTorchValueTagNone;
}

@end
