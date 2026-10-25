/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <XCTest/XCTest.h>

#import <ExecuTorch/ExecuTorch.h>

#import <stdatomic.h>

// The method ExecuTorch's own log calls go through, so tests can emit logs.
@interface ExecuTorchLog (Testing)

- (void)logWithLevel:(ExecuTorchLogLevel)level
           timestamp:(NSTimeInterval)timestamp
            filename:(NSString *)filename
                line:(NSUInteger)line
             message:(NSString *)message;

@end

@interface LogTestSink : NSObject<ExecuTorchLogSink>

@property(nonatomic, readonly) NSInteger logCount;
@property(atomic, copy, nullable) NSString *lastMessage;
@property(nonatomic, copy, nullable) void (^onLog)(LogTestSink *sink);
@property(nonatomic, copy, nullable) void (^onDealloc)(void);

@end

@implementation LogTestSink {
  atomic_long _logCount;
}

- (NSInteger)logCount {
  return atomic_load(&_logCount);
}

- (void)logWithLevel:(ExecuTorchLogLevel)level
           timestamp:(NSTimeInterval)timestamp
            filename:(NSString *)filename
                line:(NSUInteger)line
             message:(NSString *)message {
  atomic_fetch_add(&_logCount, 1);
  self.lastMessage = message;
  if (_onLog) {
    _onLog(self);
  }
}

- (void)dealloc {
  [ExecuTorchLog.sharedLog removeSink:self];
  if (_onDealloc) {
    _onDealloc();
  }
}

@end

// Every instance is equal to every other, so the log must tell them apart by
// identity.
@interface LogTestEqualSink : LogTestSink
@end

@implementation LogTestEqualSink

- (BOOL)isEqual:(id)object {
  return [object isKindOfClass:[LogTestEqualSink class]];
}

- (NSUInteger)hash {
  return 0;
}

@end

@interface LogTestObjC : XCTestCase
@end

@implementation LogTestObjC

- (void)log:(NSString *)message {
  [ExecuTorchLog.sharedLog logWithLevel:ExecuTorchLogLevelInfo
                              timestamp:0
                               filename:@(__FILE__)
                                   line:__LINE__
                                message:message];
}

- (void)testSinkRemovesItselfInDealloc {
  ExecuTorchLog *log = ExecuTorchLog.sharedLog;
  dispatch_group_t deallocated = dispatch_group_create();
  for (NSInteger index = 0; index < 3000; ++index) {
    @autoreleasepool {
      LogTestSink *sink = [LogTestSink new];
      dispatch_group_enter(deallocated);
      sink.onDealloc = ^{
        dispatch_group_leave(deallocated);
      };
      if (index % 3 != 0) {
        [log addSink:sink];
      }
      if (index % 3 == 2) {
        [log removeSink:sink];
      }
    }
  }
  XCTAssertEqual(
      dispatch_group_wait(deallocated,
                          dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)),
      0);
}

- (void)testRemovedSinkReceivesNoLaterLogs {
  ExecuTorchLog *log = ExecuTorchLog.sharedLog;
  LogTestSink *sink = [LogTestSink new];
  [log addSink:sink];
  [self log:@"before removal"];
  const NSInteger logCount = sink.logCount;
  XCTAssertGreaterThan(logCount, 0);

  [log removeSink:sink];
  [self log:@"after removal"];
  XCTAssertEqual(sink.logCount, logCount);
}

- (void)testSinkRemovesItselfDuringReplay {
  ExecuTorchLog *log = ExecuTorchLog.sharedLog;
  [self log:@"buffered"];
  XCTestExpectation *replayed = [self expectationWithDescription:@"replayed"];
  LogTestSink *sink = [LogTestSink new];
  sink.onLog = ^(LogTestSink *loggingSink) {
    if (loggingSink.logCount == 1) {
      [ExecuTorchLog.sharedLog removeSink:loggingSink];
      [replayed fulfill];
    }
  };
  [log addSink:sink];
  [self waitForExpectations:@[ replayed ] timeout:10];

  // Waits for the rest of the replay, which may still reach the sink.
  [self log:@"after replay"];
  const NSInteger logCount = sink.logCount;
  [self log:@"after removal"];
  XCTAssertEqual(sink.logCount, logCount);
  XCTAssertNotEqualObjects(sink.lastMessage, @"after replay");
}

- (void)testRemovingOneSinkKeepsTheOthers {
  ExecuTorchLog *log = ExecuTorchLog.sharedLog;
  LogTestSink *removed = [LogTestEqualSink new];
  LogTestSink *kept = [LogTestEqualSink new];
  [log addSink:removed];
  [log addSink:kept];
  [log removeSink:removed];
  [self log:@"only for the kept sink"];
  XCTAssertNotEqualObjects(removed.lastMessage, @"only for the kept sink");
  XCTAssertEqualObjects(kept.lastMessage, @"only for the kept sink");
}

- (void)testConcurrentLoggingWhileSinksComeAndGo {
  ExecuTorchLog *log = ExecuTorchLog.sharedLog;
  dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0);
  dispatch_group_t done = dispatch_group_create();
  dispatch_group_t deallocated = dispatch_group_create();
  for (NSInteger index = 0; index < 4000; ++index) {
    dispatch_group_async(done, queue, ^{
      if (index % 4 == 0) {
        [self log:@"concurrent"];
        return;
      }
      LogTestSink *sink = [LogTestSink new];
      dispatch_group_enter(deallocated);
      sink.onDealloc = ^{
        dispatch_group_leave(deallocated);
      };
      [log addSink:sink];
      if (index % 4 == 2) {
        [self log:@"concurrent"];
      }
      if (index % 4 == 3) {
        [log removeSink:sink];
      }
    });
  }
  const dispatch_time_t timeout =
      dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC);
  XCTAssertEqual(dispatch_group_wait(done, timeout), 0);
  XCTAssertEqual(dispatch_group_wait(deallocated, timeout), 0);
}

@end
