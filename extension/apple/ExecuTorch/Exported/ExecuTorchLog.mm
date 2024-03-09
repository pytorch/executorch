/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchLog.h"

#import <os/log.h>

#import <executorch/runtime/platform/platform.h>

@interface ExecuTorchLog ()

- (void)logWithLevel:(ExecuTorchLogLevel)level
           timestamp:(NSTimeInterval)timestamp
            filename:(NSString *)filename
                line:(NSUInteger)line
             message:(NSString *)message;

@end

@implementation ExecuTorchLog {
  NSHashTable<id<ExecuTorchLogSink>> *_sinks;
  dispatch_queue_t _queue;
}

+ (instancetype)sharedLog {
  static ExecuTorchLog *sharedLog;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    sharedLog = [self new];
    sharedLog->_sinks = [NSHashTable weakObjectsHashTable];
    sharedLog->_queue = dispatch_queue_create("org.pytorch.executorch.log",
                                              DISPATCH_QUEUE_SERIAL);
  });
  return sharedLog;
}

- (void)addSink:(id<ExecuTorchLogSink>)sink {
  dispatch_async(_queue, ^{
    [self->_sinks addObject:sink];
  });
}

- (void)removeSink:(id<ExecuTorchLogSink>)sink {
  dispatch_async(_queue, ^{
    [self->_sinks removeObject:sink];
  });
}

#pragma mark - Private

- (void)logWithLevel:(ExecuTorchLogLevel)level
           timestamp:(NSTimeInterval)timestamp
            filename:(NSString *)filename
                line:(NSUInteger)line
             message:(NSString *)message {
  NSHashTable<id<ExecuTorchLogSink>> __block *sinks;
  dispatch_sync(_queue, ^{
    sinks = [self->_sinks copy];
  });
  for (id<ExecuTorchLogSink> sink in sinks) {
    [sink logWithLevel:level
             timestamp:timestamp
              filename:filename
                  line:line
               message:message];
  }
}

@end

void et_pal_emit_log_message(et_timestamp_t timestamp,
                             et_pal_log_level_t level,
                             const char *__nonnull filename,
                             __ET_UNUSED const char *function,
                             size_t line,
                             const char *__nonnull message,
                             __ET_UNUSED size_t length) {
  NSTimeInterval timeInterval = timestamp / 1000000000.0;
  NSUInteger totalSeconds = (NSUInteger)timeInterval;
  NSUInteger hours = (totalSeconds / 3600) % 24;
  NSUInteger minutes = (totalSeconds / 60) % 60;
  NSUInteger seconds = totalSeconds % 60;
  NSUInteger microseconds = (timestamp - totalSeconds) * 1000000;
  NSString *formattedMessage = [NSString
      stringWithFormat:@"%c %02lu:%02lu:%02lu.%06lu executorch:%s:%zu] %s",
                       (char)level,
                       hours,
                       minutes,
                       seconds,
                       microseconds,
                       filename,
                       line,
                       message];
  os_log_type_t logType = OS_LOG_TYPE_DEFAULT;
  switch (level) {
  case kDebug:
    logType = OS_LOG_TYPE_DEBUG;
    break;
  case kInfo:
    logType = OS_LOG_TYPE_INFO;
    break;
  case kError:
    logType = OS_LOG_TYPE_ERROR;
    break;
  case kFatal:
    logType = OS_LOG_TYPE_FAULT;
    break;
  default:
    logType = OS_LOG_TYPE_DEFAULT;
    break;
  }
  os_log_with_type(OS_LOG_DEFAULT, logType, "%{public}@", formattedMessage);

  [ExecuTorchLog.sharedLog
      logWithLevel:(ExecuTorchLogLevel)level
         timestamp:timeInterval
          filename:[NSString stringWithUTF8String:filename]
              line:(NSUInteger)line
           message:[NSString stringWithUTF8String:message]];
}
