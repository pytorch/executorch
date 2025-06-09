/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchModule.h"

#import "ExecuTorchError.h"
#import "ExecuTorchUtils.h"

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

@interface ExecuTorchTensorMetadata ()

- (instancetype)initWithTensorMetadata:(const TensorInfo &)tensorInfo
    NS_DESIGNATED_INITIALIZER;

@end

@implementation ExecuTorchTensorMetadata {
  NSArray<NSNumber *> *_shape;
  NSArray<NSNumber *> *_dimensionOrder;
  ExecuTorchDataType _dataType;
  BOOL _isMemoryPlanned;
  NSString *_name;
}

- (instancetype)initWithTensorMetadata:(const TensorInfo &)tensorInfo {
  self = [super init];
  if (self) {
    _shape = utils::toNSArray(tensorInfo.sizes());
    _dimensionOrder = utils::toNSArray(tensorInfo.dim_order());
    _dataType = (ExecuTorchDataType)tensorInfo.scalar_type();
    _isMemoryPlanned = tensorInfo.is_memory_planned();
    _name = [[NSString alloc] initWithBytes:tensorInfo.name().data()
                                     length:tensorInfo.name().size()
                                   encoding:NSUTF8StringEncoding];
  }
  return self;
}

@end

@interface ExecuTorchMethodMetadata ()

- (nullable instancetype)initWithMethodMetadata:(const MethodMeta &)methodMeta
                                          error:(NSError **)error
    NS_DESIGNATED_INITIALIZER;

@end

@implementation ExecuTorchMethodMetadata {
  NSString *_name;
  NSMutableArray<NSNumber *> *_inputValueTags;
  NSMutableArray<NSNumber *> *_outputValueTags;
  NSMutableDictionary<NSNumber *, ExecuTorchTensorMetadata *> *_inputTensorMetadatas;
  NSMutableDictionary<NSNumber *, ExecuTorchTensorMetadata *> *_outputTensorMetadatas;
  NSMutableArray<ExecuTorchTensorMetadata *> *_attributeTensorMetadatas;
  NSMutableArray<NSNumber *> *_memoryPlannedBufferSizes;
  NSMutableArray<NSString *> *_backendNames;
  NSInteger _instructionCount;
}

- (nullable instancetype)initWithMethodMetadata:(const MethodMeta &)methodMeta
                                          error:(NSError **)error {
  self = [super init];
  if (self) {
    _name = @(methodMeta.name());
    const NSInteger inputCount = methodMeta.num_inputs();
    const NSInteger outputCount = methodMeta.num_outputs();
    const NSInteger attributeCount = methodMeta.num_attributes();
    const NSInteger memoryPlannedBufferCount = methodMeta.num_memory_planned_buffers();
    const NSInteger backendCount = methodMeta.num_backends();
    _instructionCount = methodMeta.num_instructions();
    _inputValueTags = [NSMutableArray arrayWithCapacity:inputCount];
    _outputValueTags = [NSMutableArray arrayWithCapacity:outputCount];
    _inputTensorMetadatas = [NSMutableDictionary dictionary];
    _outputTensorMetadatas = [NSMutableDictionary dictionary];
    _attributeTensorMetadatas = [NSMutableArray arrayWithCapacity:attributeCount];
    _memoryPlannedBufferSizes = [NSMutableArray arrayWithCapacity:memoryPlannedBufferCount];
    _backendNames = [NSMutableArray arrayWithCapacity:backendCount];

    for (NSInteger index = 0; index < inputCount; ++index) {
      auto result = methodMeta.input_tag(index);
      if (!result.ok()) {
        if (error) {
          *error = ExecuTorchErrorWithCode((ExecuTorchErrorCode)result.error());
        }
        return nil;
      }
      const auto inputValueTag = (ExecuTorchValueTag)result.get();
      [_inputValueTags addObject:@(inputValueTag)];

      if (inputValueTag == ExecuTorchValueTagTensor) {
        auto tensorMetadataResult = methodMeta.input_tensor_meta(index);
        if (!tensorMetadataResult.ok()) {
          if (error) {
            *error = ExecuTorchErrorWithCode((ExecuTorchErrorCode)tensorMetadataResult.error());
          }
          return nil;
        }
        _inputTensorMetadatas[@(index)] = [[ExecuTorchTensorMetadata alloc] initWithTensorMetadata:tensorMetadataResult.get()];
      }
    }
    for (NSInteger index = 0; index < outputCount; ++index) {
      auto result = methodMeta.output_tag(index);
      if (!result.ok()) {
        if (error) {
          *error = ExecuTorchErrorWithCode((ExecuTorchErrorCode)result.error());
        }
        return nil;
      }
      const auto outputValueTag = (ExecuTorchValueTag)result.get();
      [_outputValueTags addObject:@(outputValueTag)];

      if (outputValueTag == ExecuTorchValueTagTensor) {
        auto tensorMetadataResult = methodMeta.output_tensor_meta(index);
        if (!tensorMetadataResult.ok()) {
          if (error) {
            *error = ExecuTorchErrorWithCode((ExecuTorchErrorCode)tensorMetadataResult.error());
          }
          return nil;
        }
        _outputTensorMetadatas[@(index)] = [[ExecuTorchTensorMetadata alloc] initWithTensorMetadata:tensorMetadataResult.get()];
      }
    }
    for (NSInteger index = 0; index < attributeCount; ++index) {
      auto result = methodMeta.attribute_tensor_meta(index);
      if (!result.ok()) {
        if (error) {
          *error = ExecuTorchErrorWithCode((ExecuTorchErrorCode)result.error());
        }
        return nil;
      }
      [_attributeTensorMetadatas addObject:[[ExecuTorchTensorMetadata alloc] initWithTensorMetadata:result.get()]];
    }
    for (NSInteger index = 0; index < memoryPlannedBufferCount; ++index) {
      auto result = methodMeta.memory_planned_buffer_size(index);
      if (!result.ok()) {
        if (error) {
          *error = ExecuTorchErrorWithCode((ExecuTorchErrorCode)result.error());
        }
        return nil;
      }
      const auto memoryPlannedBufferSize = result.get();
      [_memoryPlannedBufferSizes addObject:@(memoryPlannedBufferSize)];
    }
    for (NSInteger index = 0; index < backendCount; ++index) {
      auto result = methodMeta.get_backend_name(index);
      if (!result.ok()) {
        if (error) {
          *error = ExecuTorchErrorWithCode((ExecuTorchErrorCode)result.error());
        }
        return nil;
      }
      NSString *backendName = [NSString stringWithUTF8String:result.get()];
      [_backendNames addObject:backendName];
    }
  }
  return self;
}

- (NSArray<NSNumber *> *)inputValueTags {
  return _inputValueTags;
}

- (NSArray<NSNumber *> *)outputValueTags {
  return _outputValueTags;
}

- (NSDictionary<NSNumber *,ExecuTorchTensorMetadata *> *)inputTensorMetadatas {
  return _inputTensorMetadatas;
}

- (NSDictionary<NSNumber *,ExecuTorchTensorMetadata *> *)outputTensorMetadatas {
  return _outputTensorMetadatas;
}

- (NSArray<ExecuTorchTensorMetadata *> *)attributeTensorMetadatas {
  return _attributeTensorMetadatas;
}

- (NSArray<NSNumber *> *)memoryPlannedBufferSizes {
  return _memoryPlannedBufferSizes;
}

- (NSArray<NSString *> *)backendNames {
  return _backendNames;
}

@end

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
      *error = ExecuTorchErrorWithCode((ExecuTorchErrorCode)errorCode);
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
      *error = ExecuTorchErrorWithCode((ExecuTorchErrorCode)errorCode);
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
      *error = ExecuTorchErrorWithCode((ExecuTorchErrorCode)result.error());
    }
    return nil;
  }
  NSMutableSet<NSString *> *methods = [NSMutableSet setWithCapacity:result->size()];
  for (const auto &name : *result) {
    [methods addObject:(NSString *)@(name.c_str())];
  }
  return methods;
}

- (nullable ExecuTorchMethodMetadata *)methodMetadata:(NSString *)methodName
                                                error:(NSError **)error {
  const auto result = _module->method_meta(methodName.UTF8String);
  if (!result.ok()) {
    if (error) {
      *error = ExecuTorchErrorWithCode((ExecuTorchErrorCode)result.error());
    }
    return nil;
  }
  return [[ExecuTorchMethodMetadata alloc] initWithMethodMetadata:result.get()
                                                            error:error];
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
      *error = ExecuTorchErrorWithCode((ExecuTorchErrorCode)result.error());
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
