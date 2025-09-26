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

@implementation ExecuTorchLLMTextRunner {
  NSString *_modelPath;
  NSString *_tokenizerPath;
  std::unique_ptr<std::vector<std::string>> _specialTokens;
  std::unique_ptr<llm::TextLLMRunner> _runner;
}

- (instancetype)initWithModelPath:(NSString*)modelPath
                    tokenizerPath:(NSString*)tokenizerPath
                    specialTokens:(NSArray<NSString*>*)tokens {
  self = [super init];
  if (self) {
    _modelPath = [modelPath copy];
    _tokenizerPath = [tokenizerPath copy];
    _specialTokens = std::make_unique<std::vector<std::string>>();
    for (NSString *token in tokens) {
      _specialTokens->emplace_back(token.UTF8String);
    }
  }
  return self;
}

- (BOOL)isLoaded {
  return _runner && _runner->is_loaded();
}

- (BOOL)loadWithError:(NSError**)error {
  if (![self isLoaded]) {
    _runner = llm::create_text_llm_runner(
      _modelPath.UTF8String,
      llm::load_tokenizer(_tokenizerPath.UTF8String, std::move(_specialTokens))
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

- (BOOL)generate:(NSString*)prompt
    sequenceLength:(NSInteger)seq_len
withTokenCallback:(nullable void (^)(NSString*))callback
                error:(NSError**)error {
  if (![self loadWithError:error]) {
    return NO;
  }
  auto status = _runner->generate(
    prompt.UTF8String,
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

- (void)reset {
  if (_runner) {
    _runner->reset();
  }
}

@end
