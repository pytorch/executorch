/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "LLaMARunner.h"

#import <ExecuTorch/ExecuTorchLog.h>
#if BUILD_WITH_XCODE
#import "ExecuTorchTextLLMRunner.h"
#else
#import <ExecuTorchLLM/ExecuTorchLLM.h>
#endif
#import <executorch/examples/models/llama/tokenizer/llama_tiktoken.h>

@interface LLaMARunner ()<ExecuTorchLogSink>
@end

@implementation LLaMARunner {
  ExecuTorchTextLLMRunner *_runner;
}

- (instancetype)initWithModelPath:(NSString *)modelPath
                    tokenizerPath:(NSString *)tokenizerPath {
  self = [super init];
  if (self) {
    [ExecuTorchLog.sharedLog addSink:self];
    auto tokens = example::get_special_tokens(example::Version::Default);
    NSMutableArray<NSString*> *specialTokens = [[NSMutableArray alloc] initWithCapacity:tokens->size()];
    for (const auto &token : *tokens) {
      [specialTokens addObject:(NSString *)@(token.c_str())];
    }
    _runner = [[ExecuTorchTextLLMRunner alloc] initWithModelPath:modelPath
                                                   tokenizerPath:tokenizerPath
                                                   specialTokens:specialTokens];
  }
  return self;
}

- (void)dealloc {
  [ExecuTorchLog.sharedLog removeSink:self];
}

- (BOOL)isLoaded {
  return [_runner isLoaded];
}

- (BOOL)loadWithError:(NSError**)error {
  return [_runner loadWithError:error];
}

- (BOOL)generate:(NSString *)prompt
    sequenceLength:(NSInteger)seq_len
 withTokenCallback:(nullable void (^)(NSString *))callback
             error:(NSError **)error {
  return [_runner generate:prompt
            sequenceLength:seq_len
         withTokenCallback:callback
                     error:error];
}

- (void)stop {
  [_runner stop];
}

#pragma mark - ExecuTorchLogSink

- (void)logWithLevel:(ExecuTorchLogLevel)level
           timestamp:(NSTimeInterval)timestamp
            filename:(NSString*)filename
                line:(NSUInteger)line
             message:(NSString*)message {
  NSUInteger totalSeconds = (NSUInteger)timestamp;
  NSUInteger hours = (totalSeconds / 3600) % 24;
  NSUInteger minutes = (totalSeconds / 60) % 60;
  NSUInteger seconds = totalSeconds % 60;
  NSUInteger microseconds = (timestamp - totalSeconds) * 1000000;
  NSLog(
    @"%c %02lu:%02lu:%02lu.%06lu executorch:%s:%zu] %s",
    (char)level,
    hours,
    minutes,
    seconds,
    microseconds,
    filename.UTF8String,
    line,
    message.UTF8String
  );
}

@end
