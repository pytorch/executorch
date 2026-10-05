/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ETCoreAIBridge.h"

NSErrorDomain const ETCoreAIErrorDomain = @"org.pytorch.executorch.coreai";

@implementation ETCoreAIInputTensor
- (instancetype)initWithBytes:(const void*)bytes
                    byteCount:(NSUInteger)byteCount
                        shape:(NSArray<NSNumber*>*)shape
                   scalarType:(ETCoreAIScalarType)scalarType {
  self = [super init];
  if (self) {
    _bytes = bytes;
    _byteCount = byteCount;
    _shape = [shape copy];
    _scalarType = scalarType;
  }
  return self;
}
@end

@implementation ETCoreAITensor
- (instancetype)initWithData:(NSData*)data
                       shape:(NSArray<NSNumber*>*)shape
                  scalarType:(ETCoreAIScalarType)scalarType {
  self = [super init];
  if (self) {
    _data = [data copy];
    _shape = [shape copy];
    _scalarType = scalarType;
  }
  return self;
}
@end
