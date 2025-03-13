// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

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

- (nullable NSString *)stringValueAndReturnError:(NSError * _Nullable * _Nullable)error
{
  if (error) {
    *error = [ModelRuntimeValueErrorFactory unsupportedType:@"ExecutorchRuntimeValue doesn't support strings"];
  }
  return nil;
}

- (nullable id<ModelRuntimeTensorValueBridging>)tensorValueAndReturnError:(NSError * _Nullable * _Nullable)error
{
  if (_value.isTensor()) {
    return [[ExecutorchRuntimeTensorValue alloc] initWithTensor:_value.toTensor() error:error];
  }

  if (error) {
    *error = [ModelRuntimeValueErrorFactory
              invalidType:[NSString stringWithFormat:@"Tag::%d", _value.tag]
              expectedType:@"Tag::Tensor"];
  }
  return nil;
}

- (EValue)getBackedValue
{
  return _value;
}

- (NSArray<id<ModelRuntimeValueBridging>> *)arrayValueAndReturnError:(NSError * _Nullable * _Nullable)error
{
  if (error) {
    *error = [ModelRuntimeValueErrorFactory unsupportedType:@"EValue doesn't support arrays"];
  }
  return nil;
}

@end
