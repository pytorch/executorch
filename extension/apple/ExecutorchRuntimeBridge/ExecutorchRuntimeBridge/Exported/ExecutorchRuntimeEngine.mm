/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecutorchRuntimeEngine.h"

#import <map>
#import <vector>

#import <executorch/extension/module/module.h>

@implementation ExecutorchRuntimeEngine
{
  NSString *_modelPath;
  NSString *_modelMethodName;
  std::unique_ptr<torch::executor::Module> _module;
}

- (instancetype)initWithModelPath:(NSString *)modelPath
                  modelMethodName:(NSString *)modelMethodName
                            error:(NSError **)error
{
  if (self = [super init]) {
    _modelPath = modelPath;
    _modelMethodName = modelMethodName;
    _module = std::make_unique<torch::executor::Module>(modelPath.UTF8String);
    const auto e = _module->load_method(modelMethodName.UTF8String);
    if (e != executorch::runtime::Error::Ok) {
      if (error) {
        *error = [NSError errorWithDomain:@"ExecutorchRuntimeEngine"
                                      code:(NSInteger)e
                                  userInfo:nil];
      }
      return nil;
    }
  }
  return self;
}

- (nullable NSArray<ExecutorchRuntimeValue *> *)infer:(NSArray<ExecutorchRuntimeValue *> *)values
                                                error:(NSError **)error
{
  std::vector<torch::executor::EValue> inputEValues;
  inputEValues.reserve(values.count);
  for (ExecutorchRuntimeValue *inputValue in values) {
    inputEValues.push_back([inputValue getBackedValue]);
  }
  const auto result = _module->execute(_modelMethodName.UTF8String, inputEValues);
  if (!result.ok()) {
    if (error) {
      *error = [NSError errorWithDomain:@"ExecutorchRuntimeEngine"
                                    code:(NSInteger)result.error()
                                userInfo:nil];
    }
    return nil;
  }
  NSMutableArray<ExecutorchRuntimeValue *> *const resultValues = [NSMutableArray new];
  for (const auto &evalue : result.get()) {
    [resultValues addObject:[[ExecutorchRuntimeValue alloc] initWithEValue:evalue]];
  }
  return resultValues;
}

@end
