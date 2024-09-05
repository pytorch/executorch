/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "LLaMARunner.h"

#import <ExecuTorch/ExecuTorchLog.h>
#import <executorch/examples/models/llama2/runner/runner.h>
#import <executorch/examples/models/llava/runner/llava_runner.h>

using namespace ::torch::executor;

NSErrorDomain const LLaMARunnerErrorDomain = @"LLaMARunnerErrorDomain";
NSErrorDomain const LLaVARunnerErrorDomain = @"LLaVARunnerErrorDomain";

@interface LLaMARunner ()<ExecuTorchLogSink>
@end

@implementation LLaMARunner {
  std::unique_ptr<Runner> _runner;
}

- (instancetype)initWithModelPath:(NSString*)modelPath
                    tokenizerPath:(NSString*)tokenizerPath {
  self = [super init];
  if (self) {
    [ExecuTorchLog.sharedLog addSink:self];
    _runner = std::make_unique<Runner>(
        modelPath.UTF8String, tokenizerPath.UTF8String);
  }
  return self;
}

- (void)dealloc {
  [ExecuTorchLog.sharedLog removeSink:self];
}

- (BOOL)isloaded {
  return _runner->is_loaded();
}

- (BOOL)loadWithError:(NSError**)error {
  const auto status = _runner->load();
  if (status != Error::Ok) {
    if (error) {
      *error = [NSError errorWithDomain:LLaMARunnerErrorDomain
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
  const auto status = _runner->generate(
      prompt.UTF8String, seq_len, [callback](const std::string& token) {
        callback(@(token.c_str()));
      });
  if (status != Error::Ok) {
    if (error) {
      *error = [NSError errorWithDomain:LLaMARunnerErrorDomain
                                   code:(NSInteger)status
                               userInfo:nil];
      return NO;
    }
  }
  return YES;
}

- (void)stop {
  _runner->stop();
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
      message.UTF8String);
}

@end

@interface LLaVARunner ()<ExecuTorchLogSink>
@end

@implementation LLaVARunner {
  std::unique_ptr<LlavaRunner> _runner;
}

- (instancetype)initWithModelPath:(NSString*)modelPath
                    tokenizerPath:(NSString*)tokenizerPath {
  self = [super init];
  if (self) {
    [ExecuTorchLog.sharedLog addSink:self];
    _runner = std::make_unique<LlavaRunner>(
        modelPath.UTF8String, tokenizerPath.UTF8String);
  }
  return self;
}

- (void)dealloc {
  [ExecuTorchLog.sharedLog removeSink:self];
}

- (BOOL)isloaded {
  return _runner->is_loaded();
}

- (BOOL)loadWithError:(NSError**)error {
  const auto status = _runner->load();
  if (status != Error::Ok) {
    if (error) {
      *error = [NSError errorWithDomain:LLaVARunnerErrorDomain
                                   code:(NSInteger)status
                               userInfo:nil];
    }
    return NO;
  }
  return YES;
}

- (BOOL)generate:(NSArray<UIImage*>*)images
               prompt:(NSString*)prompt
       sequenceLength:(NSInteger)seq_len
    withTokenCallback:(nullable void (^)(NSString*))callback
                error:(NSError**)error {
  std::vector<Image> rawImages;
  rawImages.reserve(images.count);

  for (UIImage* image in images) {
    CGImageRef cgImage = image.CGImage;
    const int32_t width = CGImageGetWidth(cgImage);
    const int32_t height = CGImageGetHeight(cgImage);
    std::vector<uint8_t> buffer(height * width * 4);
    CGContextRef context = CGBitmapContextCreate(
        buffer.data(),
        width,
        height,
        8,
        width * 4,
        CGColorSpaceCreateDeviceRGB(),
        kCGImageAlphaPremultipliedLast);
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), cgImage);
    CGContextRelease(context);
    rawImages.push_back({std::move(buffer), width, height, 4});
  }
  const auto status = _runner->generate(
      std::move(rawImages),
      prompt.UTF8String,
      seq_len,
      [callback](const std::string& token) { callback(@(token.c_str())); });
  if (status != Error::Ok) {
    if (error) {
      *error = [NSError errorWithDomain:LLaVARunnerErrorDomain
                                   code:(NSInteger)status
                               userInfo:nil];
      return NO;
    }
  }
  return YES;
}

- (void)stop {
  _runner->stop();
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
      message.UTF8String);
}

@end
