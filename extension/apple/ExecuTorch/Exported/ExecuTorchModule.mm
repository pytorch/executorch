/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchModule.h"

#import "ExecuTorchError.h"

#import <executorch/extension/module/module.h>
#import <executorch/extension/tensor/tensor.h>

using namespace executorch::extension;
using namespace executorch::runtime;

static inline EValue toEValue(ExecuTorchValue *value) {
  if (value.isTensor) {
    auto *nativeTensorPtr = value.tensorValue.nativeInstance;
    ET_CHECK(nativeTensorPtr);
    auto nativeTensor = *reinterpret_cast<TensorPtr *>(nativeTensorPtr);
    ET_CHECK(nativeTensor);
    return *nativeTensor;
  }
  if (value.isDouble) {
    return EValue(value.doubleValue);
  }
  if (value.isInteger) {
    return EValue(static_cast<int64_t>(value.intValue));
  }
  if (value.isBoolean) {
    return EValue(value.boolValue);
  }
  ET_CHECK_MSG(false, "Unsupported ExecuTorchValue type");
  return EValue();
}

static inline ExecuTorchValue *toExecuTorchValue(EValue value) {
  if (value.isTensor()) {
    auto nativeInstance = make_tensor_ptr(value.toTensor());
    return [ExecuTorchValue valueWithTensor:[[ExecuTorchTensor alloc] initWithNativeInstance:&nativeInstance]];
  }
  if (value.isDouble()) {
    return [ExecuTorchValue valueWithDouble:value.toDouble()];
  }
  if (value.isInt()) {
    return [ExecuTorchValue valueWithInteger:value.toInt()];
  }
  if (value.isBool()) {
    return [ExecuTorchValue valueWithBoolean:value.toBool()];
  }
  if (value.isString()) {
    const auto stringView = value.toString();
    NSString *string = [[NSString alloc] initWithBytes:stringView.data()
                                                length:stringView.size()
                                              encoding:NSUTF8StringEncoding];
    return [ExecuTorchValue valueWithString:string];
  }
  ET_CHECK_MSG(false, "Unsupported EValue type");
  return [ExecuTorchValue new];
}

@implementation ExecuTorchModule {
  std::unique_ptr<Module> _module;
}

- (instancetype)initWithFilePath:(NSString *)filePath
                        loadMode:(ExecuTorchModuleLoadMode)loadMode {
  self = [super init];
  if (self) {
    _module = std::make_unique<Module>(
      filePath.UTF8String,
      static_cast<Module::LoadMode>(loadMode)
    );
  }
  return self;
}

- (instancetype)initWithFilePath:(NSString *)filePath {
  return [self initWithFilePath:filePath loadMode:ExecuTorchModuleLoadModeFile];
}

- (BOOL)loadWithVerification:(ExecuTorchVerification)verification
                       error:(NSError **)error {
  const auto errorCode = _module->load(static_cast<Program::Verification>(verification));
  if (errorCode != Error::Ok) {
    if (error) {
      *error = [NSError errorWithDomain:ExecuTorchErrorDomain
                                   code:(NSInteger)errorCode
                               userInfo:nil];
    }
    return NO;
  }
  return YES;
}

- (BOOL)load:(NSError **)error {
  return [self loadWithVerification:ExecuTorchVerificationMinimal
                              error:error];
}

- (BOOL)isLoaded {
  return _module->is_loaded();
}

- (BOOL)loadMethod:(NSString *)methodName
             error:(NSError **)error {
  const auto errorCode = _module->load_method(methodName.UTF8String);
  if (errorCode != Error::Ok) {
    if (error) {
      *error = [NSError errorWithDomain:ExecuTorchErrorDomain
                                   code:(NSInteger)errorCode
                               userInfo:nil];
    }
    return NO;
  }
  return YES;
}

- (BOOL)isMethodLoaded:(NSString *)methodName {
  return _module->is_method_loaded(methodName.UTF8String);
}

- (nullable NSSet<NSString *> *)methodNames:(NSError **)error {
  const auto result = _module->method_names();
  if (!result.ok()) {
    if (error) {
      *error = [NSError errorWithDomain:ExecuTorchErrorDomain
                                   code:(NSInteger)result.error()
                               userInfo:nil];
    }
    return nil;
  }
  NSMutableSet<NSString *> *methods = [NSMutableSet setWithCapacity:result->size()];
  for (const auto &name : *result) {
    [methods addObject:(NSString *)@(name.c_str())];
  }
  return methods;
}

- (nullable NSArray<ExecuTorchValue *> *)executeMethod:(NSString *)methodName
                                            withInputs:(NSArray<ExecuTorchValue *> *)values
                                                 error:(NSError **)error {
  std::vector<EValue> inputs;
  inputs.reserve(values.count);
  for (ExecuTorchValue *value in values) {
    inputs.push_back(toEValue(value));
  }
  const auto result = _module->execute(methodName.UTF8String, inputs);
  if (!result.ok()) {
    if (error) {
      *error = [NSError errorWithDomain:ExecuTorchErrorDomain
                                   code:(NSInteger)result.error()
                               userInfo:nil];
    }
    return nil;
  }
  NSMutableArray<ExecuTorchValue *> *outputs = [NSMutableArray arrayWithCapacity:result->size()];
  for (const auto &value : *result) {
    [outputs addObject:toExecuTorchValue(value)];
  }
  return outputs;
}

- (nullable NSArray<ExecuTorchValue *> *)executeMethod:(NSString *)methodName
                                             withInput:(ExecuTorchValue *)value
                                                 error:(NSError **)error {
  return [self executeMethod:methodName
                  withInputs:@[value]
                       error:error];
}

- (nullable NSArray<ExecuTorchValue *> *)executeMethod:(NSString *)methodName
                                                 error:(NSError **)error {
  return [self executeMethod:methodName
                  withInputs:@[]
                       error:error];
}

- (nullable NSArray<ExecuTorchValue *> *)executeMethod:(NSString *)methodName
                                           withTensors:(NSArray<ExecuTorchTensor *> *)tensors
                                                 error:(NSError **)error {
  NSMutableArray<ExecuTorchValue *> *values = [NSMutableArray arrayWithCapacity:tensors.count];
  for (ExecuTorchTensor *tensor in tensors) {
    [values addObject:[ExecuTorchValue valueWithTensor:tensor]];
  }
  return [self executeMethod:methodName
                  withInputs:values
                       error:error];
}

- (nullable NSArray<ExecuTorchValue *> *)executeMethod:(NSString *)methodName
                                            withTensor:(ExecuTorchTensor *)tensor
                                                 error:(NSError **)error {
  return [self executeMethod:methodName
                  withInputs:@[[ExecuTorchValue valueWithTensor:tensor]]
                       error:error];
}

- (nullable NSArray<ExecuTorchValue *> *)forwardWithInputs:(NSArray<ExecuTorchValue *> *)values
                                                     error:(NSError **)error {
  return [self executeMethod:@"forward"
                  withInputs:values
                       error:error];
}

- (nullable NSArray<ExecuTorchValue *> *)forwardWithInput:(ExecuTorchValue *)value
                                                    error:(NSError **)error {
  return [self executeMethod:@"forward"
                  withInputs:@[value]
                       error:error];
}

- (nullable NSArray<ExecuTorchValue *> *)forward:(NSError **)error {
  return [self executeMethod:@"forward"
                  withInputs:@[]
                       error:error];
}

- (nullable NSArray<ExecuTorchValue *> *)forwardWithTensors:(NSArray<ExecuTorchTensor *> *)tensors
                                                      error:(NSError **)error {
  NSMutableArray<ExecuTorchValue *> *values = [NSMutableArray arrayWithCapacity:tensors.count];
  for (ExecuTorchTensor *tensor in tensors) {
    [values addObject:[ExecuTorchValue valueWithTensor:tensor]];
  }
  return [self executeMethod:@"forward"
                  withInputs:values
                       error:error];
}

- (nullable NSArray<ExecuTorchValue *> *)forwardWithTensor:(ExecuTorchTensor *)tensor
                                                     error:(NSError **)error {
  return [self executeMethod:@"forward"
                  withInputs:@[[ExecuTorchValue valueWithTensor:tensor]]
                       error:error];
}

@end
