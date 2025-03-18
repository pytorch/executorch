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

static int kInitFailed = 0;
static int kInferenceFailed = 1;

static auto NSStringToString(NSString *string) -> std::string
{
  const char *cStr = [string cStringUsingEncoding:NSUTF8StringEncoding];
  if (cStr) {
    return cStr;
  }

  NSData *data = [string dataUsingEncoding:NSUTF8StringEncoding allowLossyConversion:NO];
  return {reinterpret_cast<const char *>([data bytes]), [data length]};
}

static auto StringToNSString(const std::string &string) -> NSString *
{
  CFStringRef cfString = CFStringCreateWithBytes(
    kCFAllocatorDefault,
    reinterpret_cast<const UInt8 *>(string.c_str()),
    string.size(),
    kCFStringEncodingUTF8,
    false
  );
  return (__bridge_transfer NSString *)cfString;
}

@implementation ExecutorchRuntimeEngine
{
  NSString *_modelPath;
  NSString *_modelMethodName;
  std::unique_ptr<torch::executor::Module> _module;
}

- (instancetype)initWithModelPath:(NSString *)modelPath
                  modelMethodName:(NSString *)modelMethodName
                            error:(NSError * _Nullable * _Nullable)error
{
  if (self = [super init]) {
    _modelPath = modelPath;
    _modelMethodName = modelMethodName;
    try {
      _module = std::make_unique<torch::executor::Module>(NSStringToString(modelPath));
      const auto e = _module->load_method(NSStringToString(modelMethodName));
      if (e != executorch::runtime::Error::Ok) {
        if (error) {
          *error = [NSError errorWithDomain:@"ExecutorchRuntimeEngine"
                                       code:kInitFailed
                                   userInfo:@{NSDebugDescriptionErrorKey : StringToNSString(std::to_string(static_cast<uint32_t>(e)))}];
        }
        return nil;
      }
    } catch (...) {
      if (error) {
        *error = [NSError errorWithDomain:@"ExecutorchRuntimeEngine"
                                     code:kInitFailed
                                 userInfo:@{NSDebugDescriptionErrorKey : @"Unknown error"}];
      }
      return nil;
    }
  }
  return self;
}

- (nullable NSArray<ExecutorchRuntimeValue *> *)infer:(NSArray<ExecutorchRuntimeValue *> *)input
                                                error:(NSError * _Nullable * _Nullable)error
{
  try {
    std::vector<torch::executor::EValue> inputEValues;
    inputEValues.reserve(input.count);
    for (ExecutorchRuntimeValue *inputValue in input) {
      inputEValues.push_back([inputValue getBackedValue]);
    }
    const auto result = _module->execute(NSStringToString(_modelMethodName), inputEValues);
    if (!result.ok()) {
      const auto executorchError = static_cast<uint32_t>(result.error());
      if (error) {
        *error = [NSError errorWithDomain:@"ExecutorchRuntimeEngine"
                                     code:kInferenceFailed
                                 userInfo:@{NSDebugDescriptionErrorKey : StringToNSString(std::to_string(executorchError))}];
      }
      return nil;
    }
    NSMutableArray<ExecutorchRuntimeValue *> *const resultValues = [NSMutableArray new];
    for (const auto &evalue : result.get()) {
      [resultValues addObject:[[ExecutorchRuntimeValue alloc] initWithEValue:evalue]];
    }
    return resultValues;
  } catch (...) {
    if (error) {
      *error = [NSError errorWithDomain:@"LiteInterpreterRuntimeEngine"
                                   code:kInferenceFailed
                               userInfo:@{NSDebugDescriptionErrorKey : @"Unknown error"}];
    }
    return nil;
  }
}

@end
