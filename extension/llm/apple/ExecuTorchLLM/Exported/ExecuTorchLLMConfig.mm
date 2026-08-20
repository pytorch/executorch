/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchLLMConfig.h"

#import <executorch/extension/llm/runner/irunner.h>

using namespace executorch::extension;

@interface ExecuTorchLLMConfig ()

- (const llm::GenerationConfig &)nativeConfig;

@end

@implementation ExecuTorchLLMConfig {
  std::unique_ptr<llm::GenerationConfig> _config;
}

@dynamic echoEnabled;
@dynamic maximumNewTokens;
@dynamic warming;
@dynamic sequenceLength;
@dynamic temperature;
@dynamic bosCount;
@dynamic eosCount;

- (instancetype)init {
  if (self = [super init]) {
    _config = std::make_unique<llm::GenerationConfig>();
  }
  return self;
}

- (instancetype)initWithBlock:(NS_NOESCAPE void (^)(ExecuTorchLLMConfig *))block {
  if (self = [self init]) {
    if (block) {
      block(self);
    }
  }
  return self;
}

- (id)copyWithZone:(NSZone *)zone {
  ExecuTorchLLMConfig *config = [[[self class] allocWithZone:zone] init];
  *config->_config = *_config;
  return config;
}

- (const llm::GenerationConfig &)nativeConfig {
  return *_config;
}

- (BOOL)echoEnabled {
  return _config->echo;
}

- (void)setEchoEnabled:(BOOL)echoEnabled {
  _config->echo = echoEnabled;
}

- (NSInteger)maximumNewTokens {
  return _config->max_new_tokens;
}

- (void)setMaximumNewTokens:(NSInteger)maximumNewTokens {
  _config->max_new_tokens = (int32_t)maximumNewTokens;
}

- (BOOL)warming {
  return _config->warming;
}

- (void)setWarming:(BOOL)warming {
  _config->warming = warming;
}

- (NSInteger)sequenceLength {
  return _config->seq_len;
}

- (void)setSequenceLength:(NSInteger)sequenceLength {
  _config->seq_len = (int32_t)sequenceLength;
}

- (double)temperature {
  return _config->temperature;
}

- (void)setTemperature:(double)temperature {
  _config->temperature = (float)temperature;
}

- (NSInteger)bosCount {
  return _config->num_bos;
}

- (void)setBosCount:(NSInteger)bosCount {
  _config->num_bos = (int32_t)bosCount;
}

- (NSInteger)eosCount {
  return _config->num_eos;
}

- (void)setEosCount:(NSInteger)eosCount {
  _config->num_eos = (int32_t)eosCount;
}

@end
