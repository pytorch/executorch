/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchETDump.h"

#import "ExecuTorchETDumpError.h"

#import <executorch/devtools/etdump/etdump_flatcc.h>
#import <executorch/runtime/core/event_tracer_hooks.h>

using executorch::etdump::ETDumpGen;
using executorch::etdump::ETDumpResult;

NSErrorDomain const ExecuTorchETDumpErrorDomain = @"org.pytorch.executorch.etdump";

static NSError *ETDumpError(ExecuTorchETDumpErrorCode code, NSString *message) {
  return [NSError errorWithDomain:ExecuTorchETDumpErrorDomain
                             code:code
                         userInfo:@{NSLocalizedDescriptionKey : message}];
}

@implementation ExecuTorchETDump {
  // The module owns the tracer, so this is the only owning reference either
  // needs. Reaching the tracer through the module keeps that true for the whole
  // lifetime rather than by convention.
  ExecuTorchModule *_module;
  // Serialises taking a trace against running a method. Completing a trace ends
  // the buffer being written, so the two must not overlap, and the runtime
  // performs no synchronisation of its own.
  NSLock *_lock;
}

+ (BOOL)isAvailable {
  // Reports what this framework was compiled with, which is the same build as
  // the runtime it ships beside. A caller linking a runtime built separately
  // without tracing gets a truthful answer at read time instead, where an empty
  // trace is reported rather than silently returned.
  return executorch::runtime::internal::event_tracer_enabled();
}

- (nullable instancetype)initWithFilePath:(NSString *)filePath
                            dataFilePaths:(NSArray<NSString *> *)dataFilePaths
                                 loadMode:(ExecuTorchModuleLoadMode)loadMode
                                    error:(NSError **)error {
  if (!ExecuTorchETDump.isAvailable) {
    if (error) {
      *error = ETDumpError(
          ExecuTorchETDumpErrorCodeUnavailable,
          @"This ExecuTorch runtime was built without event tracing, so no "
          @"profile can be recorded.");
    }
    return nil;
  }
  self = [super init];
  if (self) {
    _lock = [NSLock new];
    _module = [[ExecuTorchModule alloc] initWithFilePath:filePath
                                          dataFilePaths:dataFilePaths
                                            eventTracer:std::make_unique<ETDumpGen>()
                                               loadMode:loadMode];
  }
  return self;
}

- (nullable instancetype)initWithFilePath:(NSString *)filePath
                                    error:(NSError **)error {
  return [self initWithFilePath:filePath
                  dataFilePaths:@[]
                       loadMode:ExecuTorchModuleLoadModeFile
                          error:error];
}

- (ExecuTorchModule *)module {
  return _module;
}

- (nullable NSData *)takeDataWithError:(NSError **)error {
  [_lock lock];
  ETDumpResult result = [_module takeETDumpResult];
  [_lock unlock];
  if (result.buf == nullptr || result.size == 0) {
    if (error) {
      *error = ETDumpError(
          ExecuTorchETDumpErrorCodeNoData,
          @"Nothing has been recorded since the last read. Run a method first.");
    }
    return nil;
  }
  // The tracer allocated this buffer and hands it over, so copy it into an
  // object with Cocoa's lifetime and release it on every path out.
  NSData *data = [NSData dataWithBytes:result.buf length:result.size];
  free(result.buf);
  return data;
}

- (BOOL)takeDataToFile:(NSString *)path error:(NSError **)error {
  NSData *data = [self takeDataWithError:error];
  if (data == nil) {
    return NO;
  }
  return [data writeToFile:path options:NSDataWritingAtomic error:error];
}

@end
