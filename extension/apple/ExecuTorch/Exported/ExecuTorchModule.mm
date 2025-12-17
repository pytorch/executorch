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

static inline ExecuTorchValue *toExecuTorchValue(EValue value) NS_RETURNS_RETAINED {
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
  NSMutableDictionary<NSNumber *, ExecuTorchTensorMetadata *> *_inputTensorMetadata;
  NSMutableDictionary<NSNumber *, ExecuTorchTensorMetadata *> *_outputTensorMetadata;
  NSMutableArray<ExecuTorchTensorMetadata *> *_attributeTensorMetadata;
  NSMutableArray<NSNumber *> *_memoryPlannedBufferSizes;
  NSMutableArray<NSString *> *_backendNames;
  NSInteger _instructionCount;
}

- (nullable instancetype)initWithMethodMetadata:(const MethodMeta &)methodMeta
                                          error:(NSError **)error {
  self = [super init];
  if (self) {
    _name = [[NSString alloc] initWithUTF8String:methodMeta.name()];
    const NSInteger inputCount = methodMeta.num_inputs();
    const NSInteger outputCount = methodMeta.num_outputs();
    const NSInteger attributeCount = methodMeta.num_attributes();
    const NSInteger memoryPlannedBufferCount = methodMeta.num_memory_planned_buffers();
    const NSInteger backendCount = methodMeta.num_backends();
    _instructionCount = methodMeta.num_instructions();
    _inputValueTags = [[NSMutableArray alloc] initWithCapacity:inputCount];
    _outputValueTags = [[NSMutableArray alloc] initWithCapacity:outputCount];
    _inputTensorMetadata = [NSMutableDictionary new];
    _outputTensorMetadata = [NSMutableDictionary new];
    _attributeTensorMetadata = [[NSMutableArray alloc] initWithCapacity:attributeCount];
    _memoryPlannedBufferSizes = [[NSMutableArray alloc] initWithCapacity:memoryPlannedBufferCount];
    _backendNames = [[NSMutableArray alloc] initWithCapacity:backendCount];

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
        _inputTensorMetadata[@(index)] = [[ExecuTorchTensorMetadata alloc] initWithTensorMetadata:tensorMetadataResult.get()];
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
        _outputTensorMetadata[@(index)] = [[ExecuTorchTensorMetadata alloc] initWithTensorMetadata:tensorMetadataResult.get()];
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
      [_attributeTensorMetadata addObject:[[ExecuTorchTensorMetadata alloc] initWithTensorMetadata:result.get()]];
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
      NSString *backendName = [[NSString alloc] initWithUTF8String:result.get()];
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

- (NSDictionary<NSNumber *,ExecuTorchTensorMetadata *> *)inputTensorMetadata {
  return _inputTensorMetadata;
}

- (NSDictionary<NSNumber *,ExecuTorchTensorMetadata *> *)outputTensorMetadata {
  return _outputTensorMetadata;
}

- (NSArray<ExecuTorchTensorMetadata *> *)attributeTensorMetadata {
  return _attributeTensorMetadata;
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
  NSMutableDictionary<NSString *, NSMutableArray<ExecuTorchValue *> *> *_inputs;
  NSMutableDictionary<NSString *, NSMutableArray<ExecuTorchValue *> *> *_outputs;
}

- (instancetype)initWithFilePath:(NSString *)filePath
                   dataFilePaths:(NSArray<NSString *> *)dataFilePaths
                        loadMode:(ExecuTorchModuleLoadMode)loadMode {
  self = [super init];
  if (self) {
    // Convert NSArray<NSString *> to std::vector<std::string>
    std::vector<std::string> dataFilePathsVector;
    if (dataFilePaths != nil) {
      for (NSString *dataFile in dataFilePaths) {
        dataFilePathsVector.emplace_back(dataFile.UTF8String);
      }
    }
    _module = std::make_unique<Module>(
      filePath.UTF8String,
      dataFilePathsVector,
      static_cast<Module::LoadMode>(loadMode)
    );
    _inputs = [NSMutableDictionary new];
    _outputs = [NSMutableDictionary new];
  }
  return self;
}

- (instancetype)initWithFilePath:(NSString *)filePath
                   dataFilePaths:(NSArray<NSString *> *)dataFilePaths {
  return [self initWithFilePath:filePath
                  dataFilePaths:dataFilePaths
                       loadMode:ExecuTorchModuleLoadModeFile];
}

- (instancetype)initWithFilePath:(NSString *)filePath
                        loadMode:(ExecuTorchModuleLoadMode)loadMode {
  return [self initWithFilePath:filePath
                  dataFilePaths:@[]
                       loadMode:loadMode];
}
- (instancetype)initWithFilePath:(NSString *)filePath {
  return [self initWithFilePath:filePath
                  dataFilePaths:@[]
                       loadMode:ExecuTorchModuleLoadModeFile];
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

- (BOOL)unloadMethod:(NSString *)methodName {
  const auto didUnload = _module->unload_method(methodName.UTF8String);
  [_inputs removeObjectForKey:methodName];
  [_outputs removeObjectForKey:methodName];
  return didUnload;
}

- (nullable NSSet<NSString *> *)methodNames:(NSError **)error {
  const auto result = _module->method_names();
  if (!result.ok()) {
    if (error) {
      *error = ExecuTorchErrorWithCode((ExecuTorchErrorCode)result.error());
    }
    return nil;
  }
  NSMutableSet<NSString *> *methods = [[NSMutableSet alloc] initWithCapacity:result->size()];
  for (const auto &name : *result) {
    [methods addObject:(NSString *)[[NSString alloc] initWithUTF8String:name.c_str()]];
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
  const char *methodNameString = methodName.UTF8String;
  __block auto errorCode = Error::Ok;
  [values enumerateObjectsUsingBlock:^(ExecuTorchValue *value, NSUInteger index, BOOL *stop) {
    errorCode = _module->set_input(methodNameString, toEValue(value), index);
    if (errorCode != Error::Ok) {
      *stop = YES;
    }
  }];
  if (errorCode != Error::Ok) {
    if (error) {
      *error = ExecuTorchErrorWithCode((ExecuTorchErrorCode)errorCode);
    }
    return nil;
  }
  const auto result = _module->execute(methodNameString);
  if (!result.ok()) {
    if (error) {
      *error = ExecuTorchErrorWithCode((ExecuTorchErrorCode)result.error());
    }
    return nil;
  }
  NSMutableArray<ExecuTorchValue *> *outputs = [[NSMutableArray alloc] initWithCapacity:result->size()];
  for (const auto &value : *result) {
    [outputs addObject:toExecuTorchValue(value)];
  }
  return outputs;
}

- (nullable NSArray<ExecuTorchValue *> *)executeMethod:(NSString *)methodName
                                             withInput:(ExecuTorchValue *)value
                                                 error:(NSError **)error {
  return [self executeMethod:methodName
                  withInputs:[[NSArray alloc] initWithObjects:value, nil]
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
  NSMutableArray<ExecuTorchValue *> *values = [[NSMutableArray alloc] initWithCapacity:tensors.count];
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
                  withInputs:[[NSArray alloc] initWithObjects:[ExecuTorchValue valueWithTensor:tensor], nil]
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
                  withInputs:[[NSArray alloc] initWithObjects:value, nil]
                       error:error];
}

- (nullable NSArray<ExecuTorchValue *> *)forward:(NSError **)error {
  return [self executeMethod:@"forward"
                  withInputs:@[]
                       error:error];
}

- (nullable NSArray<ExecuTorchValue *> *)forwardWithTensors:(NSArray<ExecuTorchTensor *> *)tensors
                                                      error:(NSError **)error {
  NSMutableArray<ExecuTorchValue *> *values = [[NSMutableArray alloc] initWithCapacity:tensors.count];
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
                  withInputs:[[NSArray alloc] initWithObjects:[ExecuTorchValue valueWithTensor:tensor], nil]
                       error:error];
}

- (BOOL)setInput:(ExecuTorchValue *)value
           error:(NSError **)error NS_SWIFT_NAME(setInput(_:)) {
  return [self setInput:value
              forMethod:@"forward"
                atIndex:0
                  error:error];
}

- (BOOL)setInput:(ExecuTorchValue *)value
         atIndex:(NSInteger)index
           error:(NSError **)error {
  return [self setInput:value
              forMethod:@"forward"
                atIndex:index
                  error:error];
}

- (BOOL)setInput:(ExecuTorchValue *)value
       forMethod:(NSString *)methodName
           error:(NSError **)error {
  return [self setInput:value
              forMethod:methodName
                atIndex:0
                  error:error];
}

- (BOOL)setInput:(ExecuTorchValue *)value
       forMethod:(NSString *)methodName
         atIndex:(NSInteger)index
           error:(NSError **)error {
  const auto errorCode = _module->set_input(methodName.UTF8String, toEValue(value), index);
  if (errorCode != Error::Ok) {
    if (error) {
      *error = ExecuTorchErrorWithCode((ExecuTorchErrorCode)errorCode);
    }
    return NO;
  }
  // Cache inputs to keep them alive since ExecuTorchValue owns the actual data.
  NSMutableArray<ExecuTorchValue *> *inputs = _inputs[methodName];
  if (!inputs) {
    inputs = [NSMutableArray new];
    _inputs[methodName] = inputs;
  }
  if (index >= inputs.count) {
    id placeholder = NSNull.null;
    while (inputs.count < index) {
      [inputs addObject:placeholder];
    }
    [inputs addObject:value];
  } else {
    inputs[index] = value;
  }
  return YES;
}

- (BOOL)setInputs:(NSArray<ExecuTorchValue *> *)values
            error:(NSError **)error {
  return [self setInputs:values
               forMethod:@"forward"
                   error:error];
}

- (BOOL)setInputs:(NSArray<ExecuTorchValue *> *)values
        forMethod:(NSString *)methodName
            error:(NSError **)error {
  std::vector<EValue> inputs;
  inputs.reserve(values.count);
  for (ExecuTorchValue *value in values) {
    inputs.push_back(toEValue(value));
  }
  const auto errorCode = _module->set_inputs(methodName.UTF8String, inputs);
  if (errorCode != Error::Ok) {
    if (error) {
      *error = ExecuTorchErrorWithCode((ExecuTorchErrorCode)errorCode);
    }
    return NO;
  }
  // Cache inputs to keep them alive since ExecuTorchValue owns the actual data.
  _inputs[methodName] = [values mutableCopy];

  return YES;
}

- (BOOL)setOutput:(ExecuTorchValue *)value
            error:(NSError **)error {
  return [self setOutput:value
               forMethod:@"forward"
                 atIndex:0
                   error:error];
}

- (BOOL)setOutput:(ExecuTorchValue *)value
          atIndex:(NSInteger)index
            error:(NSError **)error {
  return [self setOutput:value
               forMethod:@"forward"
                 atIndex:index
                   error:error];
}

- (BOOL)setOutput:(ExecuTorchValue *)value
        forMethod:(NSString *)methodName
            error:(NSError **)error {
  return [self setOutput:value
               forMethod:methodName
                 atIndex:0
                   error:error];
}

- (BOOL)setOutput:(ExecuTorchValue *)value
        forMethod:(NSString *)methodName
          atIndex:(NSInteger)index
            error:(NSError **)error {
  const auto errorCode = _module->set_output(methodName.UTF8String, toEValue(value), index);
  if (errorCode != Error::Ok) {
    if (error) {
      *error = ExecuTorchErrorWithCode((ExecuTorchErrorCode)errorCode);
    }
    return NO;
  }
  // Cache outputs to keep them alive since ExecuTorchValue owns the actual data.
  NSMutableArray<ExecuTorchValue *> *outputs = _outputs[methodName];
  if (!outputs) {
    outputs = [NSMutableArray new];
    _outputs[methodName] = outputs;
  }
  if (index >= outputs.count) {
    id placeholder = NSNull.null;
    while (outputs.count < index) {
      [outputs addObject:placeholder];
    }
    [outputs addObject:value];
  } else {
    outputs[index] = value;
  }
  return YES;
}

@end
