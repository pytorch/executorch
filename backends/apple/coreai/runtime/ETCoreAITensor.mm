/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>
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

@implementation ETCoreAITensor {
  void (^_copyBytes)(void*);
}
- (instancetype)initWithByteCount:(NSUInteger)byteCount
                            shape:(NSArray<NSNumber*>*)shape
                       scalarType:(ETCoreAIScalarType)scalarType
                        copyBytes:(void (^)(void*))copyBytes {
  NSParameterAssert(copyBytes != nil);
  self = [super init];
  if (self) {
    _byteCount = byteCount;
    _shape = [shape copy];
    _scalarType = scalarType;
    _copyBytes = [copyBytes copy];
  }
  return self;
}
- (instancetype)initWithData:(NSData*)data
                       shape:(NSArray<NSNumber*>*)shape
                  scalarType:(ETCoreAIScalarType)scalarType {
  NSData* storage = [data copy];
  return [self initWithByteCount:storage.length
                           shape:shape
                      scalarType:scalarType
                       copyBytes:^(void* bytes) {
                         std::memcpy(bytes, storage.bytes, storage.length);
                       }];
}
- (void)copyBytesTo:(void*)bytes {
  if (_byteCount > 0) {
    _copyBytes(bytes);
  }
}
@end
