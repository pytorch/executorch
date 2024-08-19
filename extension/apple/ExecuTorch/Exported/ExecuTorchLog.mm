/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchLog.h"

#import <os/log.h>

#import <executorch/runtime/platform/log.h>
#import <executorch/runtime/platform/platform.h>

@interface ExecuTorchLog ()

- (void)logWithLevel:(ExecuTorchLogLevel)level
           timestamp:(NSTimeInterval)timestamp
            filename:(NSString *)filename
                line:(NSUInteger)line
             message:(NSString *)message;

@end

@implementation ExecuTorchLog {
#ifdef ET_LOG_ENABLED
  NSHashTable<id<ExecuTorchLogSink>> *_sinks;
  dispatch_queue_t _queue;
  NSMutableArray<NSDictionary *> *_buffer;
#endif
}

+ (instancetype)sharedLog {
  static ExecuTorchLog *sharedLog;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    sharedLog = [self new];
#if ET_LOG_ENABLED
    sharedLog->_sinks = [NSHashTable weakObjectsHashTable];
    sharedLog->_queue = dispatch_queue_create("org.pytorch.executorch.log",
                                              DISPATCH_QUEUE_SERIAL);
    sharedLog->_buffer = [NSMutableArray new];
#endif
  });
  return sharedLog;
}

- (void)addSink:(id<ExecuTorchLogSink>)sink {
#if ET_LOG_ENABLED
  dispatch_async(_queue, ^{
    [self->_sinks addObject:sink];
    for (NSDictionary *log in self->_buffer) {
      [sink logWithLevel:(ExecuTorchLogLevel)[log[@"level"] integerValue]
               timestamp:[log[@"timestamp"] doubleValue]
                filename:log[@"filename"] ?: @""
                    line:[log[@"line"] unsignedIntegerValue]
                 message:log[@"message"] ?: @""];
    }
  });
#else
  (void)sink;
#endif
}

- (void)removeSink:(id<ExecuTorchLogSink>)sink {
#if ET_LOG_ENABLED
  dispatch_async(_queue, ^{
    [self->_sinks removeObject:sink];
  });
#else
  (void)sink;
#endif
}

#pragma mark - Private

- (void)logWithLevel:(ExecuTorchLogLevel)level
           timestamp:(NSTimeInterval)timestamp
            filename:(NSString *)filename
                line:(NSUInteger)line
             message:(NSString *)message {
#if ET_LOG_ENABLED
  NSHashTable<id<ExecuTorchLogSink>> __block *sinks;
  dispatch_sync(_queue, ^{
    sinks = [self->_sinks copy];
    if (self->_buffer.count >= 100) {
      [self->_buffer removeObjectAtIndex:0];
    }
    [self->_buffer addObject:@{
      @"level" : @(level),
      @"timestamp" : @(timestamp),
      @"filename" : filename,
      @"line" : @(line),
      @"message" : message
    }];
  });
  for (id<ExecuTorchLogSink> sink in sinks) {
    [sink logWithLevel:level
             timestamp:timestamp
              filename:filename
                  line:line
               message:message];
  }
#else
  (void)level;
  (void)timestamp;
  (void)filename;
  (void)line;
  (void)message;
#endif
}

@end

void et_pal_emit_log_message(et_timestamp_t timestamp,
                             et_pal_log_level_t level,
                             const char *__nonnull filename,
                             ET_UNUSED const char *function,
                             size_t line,
                             const char *__nonnull message,
                             ET_UNUSED size_t length) {
#if ET_LOG_ENABLED
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
#else
  (void)timestamp;
  (void)level;
  (void)filename;
  (void)function;
  (void)line;
  (void)message;
  (void)length;
#endif
}
