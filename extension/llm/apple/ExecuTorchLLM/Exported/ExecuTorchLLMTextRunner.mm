/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchLLMTextRunner.h"

#import "ExecuTorchLLMError.h"

#import <executorch/extension/llm/runner/text_llm_runner.h>

using namespace executorch::extension;
using namespace executorch::runtime;

@interface ExecuTorchLLMConfig ()

- (const llm::GenerationConfig &)nativeConfig;

@end

@implementation ExecuTorchLLMTextRunner {
  NSString *_modelPath;
  NSString *_tokenizerPath;
  std::unique_ptr<std::vector<std::string>> _specialTokens;
  std::unique_ptr<llm::TextLLMRunner> _runner;
}

- (instancetype)initWithModelPath:(NSString*)modelPath
                    tokenizerPath:(NSString*)tokenizerPath {
  return [self initWithModelPath:modelPath
                   tokenizerPath:tokenizerPath
                   specialTokens:@[]];
}

- (instancetype)initWithModelPath:(NSString*)modelPath
                    tokenizerPath:(NSString*)tokenizerPath
                    specialTokens:(NSArray<NSString*>*)specialTokens {
  self = [super init];
  if (self) {
    _modelPath = [modelPath copy];
    _tokenizerPath = [tokenizerPath copy];
    _specialTokens = std::make_unique<std::vector<std::string>>();
    for (NSString *token in specialTokens) {
      _specialTokens->emplace_back(token.UTF8String ?: "");
    }
  }
  return self;
}

- (BOOL)isLoaded {
  return _runner && _runner->is_loaded();
}

- (BOOL)loadWithError:(NSError**)error {
  if (![self isLoaded]) {
    // The loader pins the begin of sequence index at 0 and the end of sequence
    // index at 1, so one entry cannot name both. Caught here because the loader
    // reports it as a tokenizer that would not load, which hides the cause.
    if (_specialTokens->size() == 1) {
      if (error) {
        *error = [NSError errorWithDomain:ExecuTorchLLMErrorDomain
                                     code:-1
                                 userInfo:@{NSLocalizedDescriptionKey: @"Special tokens must name a begin and an end of sequence token, or be empty"}];
      }
      return NO;
    }
    // An empty list means the caller had none to give, not that the tokenizer
    // should have none. Copy rather than move so a retry after a failed load
    // still has them.
    std::unique_ptr<std::vector<std::string>> specialTokens;
    if (!_specialTokens->empty()) {
      specialTokens =
        std::make_unique<std::vector<std::string>>(*_specialTokens);
    }
    _runner = llm::create_text_llm_runner(
      _modelPath.UTF8String ?: "",
      llm::load_tokenizer(
        _tokenizerPath.UTF8String ?: "",
        std::move(specialTokens)
      )
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

- (BOOL)generateWithPrompt:(NSString*)prompt
                    config:(ExecuTorchLLMConfig *)config
             tokenCallback:(nullable void (^)(NSString*))callback
                     error:(NSError**)error {
  if (![self loadWithError:error]) {
    return NO;
  }
  auto status = _runner->generate(
    prompt.UTF8String ?: "",
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
