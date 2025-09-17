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

@implementation ExecuTorchLLMImage

- (instancetype)initWithData:(NSData *)data
                       width:(NSInteger)width
                      height:(NSInteger)height
                    channels:(NSInteger)channels {
  if (self = [super init]) {
    _data = [data copy];
    _width = width;
    _height = height;
    _channels = channels;
  }
  return self;
}

- (id)copyWithZone:(NSZone *)zone {
  return self;
}

@end

@implementation ExecuTorchLLMAudio

- (instancetype)initWithData:(NSData *)data
                   batchSize:(NSInteger)batchSize
                        bins:(NSInteger)bins
                      frames:(NSInteger)frames {
  if (self = [super init]) {
    _data = [data copy];
    _batchSize = batchSize;
    _bins = bins;
    _frames = frames;
  }
  return self;
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

- (BOOL)generate:(NSArray<ExecuTorchLLMMultimodalInput *> *)inputs
   sequenceLength:(NSInteger)seq_len
withTokenCallback:(nullable void (^)(NSString *))callback
            error:(NSError **)error {
  if (![self loadWithError:error]) {
    return NO;
  }
  std::vector<llm::MultimodalInput> nativeInputs;
  for (ExecuTorchLLMMultimodalInput *input in inputs) {
    switch (input.type) {
      case ExecuTorchLLMMultimodalInputTypeText:
        nativeInputs.emplace_back(llm::MultimodalInput(input.text.UTF8String));
        break;
      case ExecuTorchLLMMultimodalInputTypeImage: {
        ExecuTorchLLMImage *image = input.image;
        std::vector<uint8_t> data((uint8_t *)image.data.bytes, (uint8_t *)image.data.bytes + image.data.length);
        nativeInputs.emplace_back(llm::MultimodalInput(llm::Image(
          std::move(data),
          (int32_t)image.width,
          (int32_t)image.height,
          (int32_t)image.channels
        )));
        break;
      }
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
    llm::GenerationConfig{.seq_len = static_cast<int32_t>(seq_len)},
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

@end
