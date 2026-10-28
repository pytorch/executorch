/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * Defines log levels with specific character codes representing each level.
 */
typedef NS_ENUM(NSInteger, ExecuTorchLogLevel) {
  ExecuTorchLogLevelDebug = 'D',
  ExecuTorchLogLevelInfo = 'I',
  ExecuTorchLogLevelError = 'E',
  ExecuTorchLogLevelFatal = 'F',
  ExecuTorchLogLevelUnknown = '?'
} NS_SWIFT_NAME(LogLevel);

/**
 * A protocol defining the requirements for a log sink to receive log messages.
 */
NS_SWIFT_NAME(LogSink)
@protocol ExecuTorchLogSink

/**
 * Logs a message with the specified additional info.
 *
 * @param level The log level of the message.
 * @param timestamp The timestamp of the log message since ExecuTorch PAL start.
 * @param filename The name of the file generating the log message.
 * @param line The line number in the file where the log message was generated.
 * @param message The log message text.
 */
- (void)logWithLevel:(ExecuTorchLogLevel)level
           timestamp:(NSTimeInterval)timestamp
            filename:(NSString *)filename
                line:(NSUInteger)line
             message:(NSString *)message
    NS_SWIFT_NAME(log(level:timestamp:filename:line:message:));

@end

/**
 * A singleton class for managing log sinks and dispatching log messages.
 */
NS_SWIFT_NAME(Log)
__attribute__((objc_subclassing_restricted))
@interface ExecuTorchLog : NSObject

/// The shared singleton log instance.
@property(class, readonly) ExecuTorchLog *sharedLog;

/**
 * Adds a log sink to receive log messages.
 *
 * The log holds the sink weakly and tells sinks apart by identity. A new sink
 * first receives the recent buffered logs.
 *
 * @param sink The log sink to add.
 */
- (void)addSink:(id<ExecuTorchLogSink>)sink NS_SWIFT_NAME(add(sink:));

/**
 * Removes a previously added log sink.
 *
 * Logs emitted after this method returns do not reach the sink. Logs emitted
 * before it returns may still reach it, including the buffered logs that
 * `addSink:` replays. Safe to call from the sink's `dealloc` or `deinit`, and
 * from its log callback.
 *
 * @param sink The log sink to remove.
 */
- (void)removeSink:(id<ExecuTorchLogSink>)sink NS_SWIFT_NAME(remove(sink:));

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
