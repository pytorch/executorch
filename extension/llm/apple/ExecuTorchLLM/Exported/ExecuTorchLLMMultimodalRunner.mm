/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchLLMMultimodalRunner.h"

#import "ExecuTorchLLMError.h"

#import <executorch/extension/llm/runner/multimodal_runner.h>

using namespace executorch::extension;
using namespace executorch::runtime;

@interface ExecuTorchLLMConfig ()

- (const llm::GenerationConfig &)nativeConfig;

@end

@implementation ExecuTorchLLMImage {
  ExecuTorchTensor *_tensor;
}

- (instancetype)initWithTensor:(ExecuTorchTensor *)tensor {
  ET_CHECK(tensor);
  if (self = [super init]) {
    ET_CHECK_MSG(tensor.shape.count == 3, "Image tensor must be rank-3 {C,H,W}");
    ExecuTorchDataType dataType = tensor.dataType;
    ET_CHECK_MSG(dataType == ExecuTorchDataTypeByte || dataType == ExecuTorchDataTypeFloat,
                 "Image tensor must be Byte or Float");
    _tensor = tensor;
  }
  return self;
}

- (instancetype)initWithData:(NSData *)data
                       width:(NSInteger)width
                      height:(NSInteger)height
                    channels:(NSInteger)channels {
  return [self initWithTensor:[[ExecuTorchTensor alloc]
                                 initWithData:data
                                        shape:@[@(channels), @(height), @(width)]
                                      dataType:ExecuTorchDataTypeByte]];
}

- (instancetype)initWithFloatData:(NSData *)data
                            width:(NSInteger)width
                           height:(NSInteger)height
                         channels:(NSInteger)channels {
  return [self initWithTensor:[[ExecuTorchTensor alloc]
                                 initWithData:data
                                        shape:@[@(channels), @(height), @(width)]
                                      dataType:ExecuTorchDataTypeFloat]];
}

- (NSInteger)width {
  return _tensor.shape[2].integerValue;
}

- (NSInteger)height {
  return _tensor.shape[1].integerValue;
}

- (NSInteger)channels {
  return _tensor.shape[0].integerValue;
}

- (BOOL)isFloat {
  return _tensor.dataType == ExecuTorchDataTypeFloat;
}

- (ExecuTorchTensor *)tensor {
  return _tensor;
}

- (id)copyWithZone:(NSZone *)zone {
  return self;
}

@end

@implementation ExecuTorchLLMAudio {
  ExecuTorchTensor *_tensor;
}

- (instancetype)initWithTensor:(ExecuTorchTensor *)tensor {
  ET_CHECK(tensor);
  if (self = [super init]) {
    ET_CHECK_MSG(tensor.shape.count == 3, "Audio tensor must be rank-3 {B,bins,frames}");
    ExecuTorchDataType dataType = tensor.dataType;
    ET_CHECK_MSG(dataType == ExecuTorchDataTypeByte || dataType == ExecuTorchDataTypeFloat,
                 "Audio tensor must be Byte or Float");
    _tensor = tensor;
  }
  return self;
}

- (instancetype)initWithData:(NSData *)data
                   batchSize:(NSInteger)batchSize
                        bins:(NSInteger)bins
                      frames:(NSInteger)frames {
  return [self initWithTensor:
      [[ExecuTorchTensor alloc] initWithData:data
                                       shape:@[@(batchSize), @(bins), @(frames)]
                                    dataType:ExecuTorchDataTypeByte]];
}

- (instancetype)initWithFloatData:(NSData *)data
                        batchSize:(NSInteger)batchSize
                             bins:(NSInteger)bins
                           frames:(NSInteger)frames {
  return [self initWithTensor:
      [[ExecuTorchTensor alloc] initWithData:data
                                       shape:@[@(batchSize), @(bins), @(frames)]
                                    dataType:ExecuTorchDataTypeFloat]];
}

- (NSInteger)batchSize {
  return _tensor.shape[0].integerValue;
}

- (NSInteger)bins {
  return _tensor.shape[1].integerValue;
}

- (NSInteger)frames {
  return _tensor.shape[2].integerValue;
}

- (BOOL)isFloat {
  return _tensor.dataType == ExecuTorchDataTypeFloat;
}

- (ExecuTorchTensor *)tensor {
  return _tensor;
}

- (id)copyWithZone:(NSZone *)zone {
  return self;
}

@end

@interface ExecuTorchLLMMultimodalInput ()

- (instancetype)initWithType:(ExecuTorchLLMMultimodalInputType)type
                        text:(NSString * __nullable)text
                       image:(ExecuTorchLLMImage * __nullable)image
                       audio:(ExecuTorchLLMAudio * __nullable)audio
    NS_DESIGNATED_INITIALIZER;

@end

@implementation ExecuTorchLLMMultimodalInput

+ (instancetype)inputWithText:(NSString *)text {
  return [[self alloc] initWithType:ExecuTorchLLMMultimodalInputTypeText
                               text:text
                              image:nil
                              audio:nil];
}

+ (instancetype)inputWithImage:(ExecuTorchLLMImage *)image {
  return [[self alloc] initWithType:ExecuTorchLLMMultimodalInputTypeImage
                               text:nil
                              image:image
                              audio:nil];
}

+ (instancetype)inputWithAudio:(ExecuTorchLLMAudio *)audio {
  return [[self alloc] initWithType:ExecuTorchLLMMultimodalInputTypeAudio
                               text:nil
                              image:nil
                              audio:audio];
}

- (instancetype)initWithType:(ExecuTorchLLMMultimodalInputType)type
                        text:(NSString * __nullable)text
                       image:(ExecuTorchLLMImage * __nullable)image
                       audio:(ExecuTorchLLMAudio * __nullable)audio {
  if (self = [super init]) {
    _type = type;
    _text = [text copy];
    _image = image;
    _audio = audio;
  }
  return self;
}

- (id)copyWithZone:(NSZone *)zone {
  return self;
}

@end

@implementation ExecuTorchLLMMultimodalRunner {
  NSString *_modelPath;
  NSString *_tokenizerPath;
  std::unique_ptr<llm::MultimodalRunner> _runner;
}

- (instancetype)initWithModelPath:(NSString*)modelPath
                    tokenizerPath:(NSString*)tokenizerPath {
  self = [super init];
  if (self) {
    _modelPath = [modelPath copy];
    _tokenizerPath = [tokenizerPath copy];
  }
  return self;
}

- (BOOL)isLoaded {
  return _runner && _runner->is_loaded();
}

- (BOOL)loadWithError:(NSError**)error {
  if (![self isLoaded]) {
    _runner = llm::create_multimodal_runner(
      _modelPath.UTF8String,
      llm::load_tokenizer(_tokenizerPath.UTF8String)
    );
    if (!_runner) {
      if (error) {
        *error = [NSError errorWithDomain:ExecuTorchLLMErrorDomain
                                     code:-1
                                 userInfo:@{NSLocalizedDescriptionKey: @"Failed to create runner"}];
      }
      return NO;
    }
  }
  auto status = _runner->load();
  if (status != Error::Ok) {
    if (error) {
      *error = [NSError errorWithDomain:ExecuTorchLLMErrorDomain
                                   code:(NSInteger)status
                               userInfo:nil];
    }
    return NO;
  }
  return YES;
}

- (BOOL)generateWithInputs:(NSArray<ExecuTorchLLMMultimodalInput *> *)inputs
                    config:(ExecuTorchLLMConfig *)config
             tokenCallback:(nullable void (^)(NSString *))callback
                     error:(NSError **)error {
  if (![self loadWithError:error]) {
    return NO;
  }
  std::vector<llm::MultimodalInput> nativeInputs;
  nativeInputs.reserve((size_t)inputs.count);
  for (ExecuTorchLLMMultimodalInput *input in inputs) {
    switch (input.type) {
      case ExecuTorchLLMMultimodalInputTypeText:
        nativeInputs.emplace_back(llm::MultimodalInput(input.text.UTF8String));
        break;
      case ExecuTorchLLMMultimodalInputTypeImage:
        nativeInputs.emplace_back(llm::MultimodalInput(llm::Image(
          make_tensor_ptr(*reinterpret_cast<TensorPtr *>(input.image.tensor.nativeInstance))
        )));
        break;
      case ExecuTorchLLMMultimodalInputTypeAudio:
        nativeInputs.emplace_back(llm::MultimodalInput(llm::Audio(
          make_tensor_ptr(*reinterpret_cast<TensorPtr *>(input.audio.tensor.nativeInstance))
        )));
        break;
      default: {
        if (error) {
          *error = [NSError errorWithDomain:ExecuTorchLLMErrorDomain
                                       code:-2
                                   userInfo:@{NSLocalizedDescriptionKey: @"Failed to create input"}];
        }
        return NO;
      }
    }
  }
  auto status = _runner->generate(
    std::move(nativeInputs),
    config.nativeConfig,
    [callback](const std::string& token) {
      if (callback) {
        callback(@(token.c_str()));
      }
    }
  );
  if (status != Error::Ok) {
    if (error) {
      *error = [NSError errorWithDomain:ExecuTorchLLMErrorDomain
                                   code:(NSInteger)status
                               userInfo:nil];
    }
    return NO;
  }
  return YES;
}

- (void)stop {
  if (_runner) {
    _runner->stop();
  }
}

- (void)reset {
  if (_runner) {
    _runner->reset();
  }
}

@end
