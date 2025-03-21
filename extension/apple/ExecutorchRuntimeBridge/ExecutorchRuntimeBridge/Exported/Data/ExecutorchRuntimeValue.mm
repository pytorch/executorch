/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecutorchRuntimeValue.h"

#import <map>
#import <vector>

#import "ExecutorchRuntimeTensorValue.h"

using torch::executor::EValue;

@implementation ExecutorchRuntimeValue
{
  EValue _value;
  // IMPORTANT
  // Tensor value keeps a reference to the original tensor value. However, the value that is wrapped by LiteInterpreterRuntimeTensorValue DOES NOT TAKE OWNERSHIP OF THE RAW DATA!
  // This means once the wrapper is deallocated, the tensor value will be deallocated as well.
  // This reference here is to keep the tensor value alive until the runtime is deallocated.
  ExecutorchRuntimeTensorValue *_tensorValue;
}

- (instancetype)initWithEValue:(EValue)value
{
  if (self = [super init]) {
    _value = value;
  }
  return self;
}

- (instancetype)initWithTensor:(ExecutorchRuntimeTensorValue *)tensorValue
{
  if (self = [self initWithEValue:EValue([tensorValue backedValue])]) {
    _tensorValue = tensorValue;
  }
  return self;
}

- (nullable ExecutorchRuntimeTensorValue *)asTensorValueAndReturnError:(NSError * _Nullable * _Nullable)error
{
  if (_value.isTensor()) {
    return [[ExecutorchRuntimeTensorValue alloc] initWithTensor:_value.toTensor() error:error];
  }

  if (error) {
    *error = [NSError
      errorWithDomain:@"ExecutorchRuntimeEngine"
      code:static_cast<uint32_t>(executorch::runtime::Error::InvalidArgument)
      userInfo: @{NSDebugDescriptionErrorKey: [NSString stringWithFormat:@"Invalid type: Tag::%d, expected Tag::Tensor", _value.tag]}];
  }
  return nil;
}

- (EValue)getBackedValue
{
  return _value;
}

@end
