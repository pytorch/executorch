/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchDump.h"

#import "ExecuTorchDumpError.h"

#import <executorch/devtools/etdump/etdump_flatcc.h>
#import <executorch/runtime/core/event_tracer_hooks.h>

#import <ExecuTorch/ExecuTorchEventTracer+Internal.h>

#import <flatcc/flatcc_builder.h>

using namespace executorch::etdump;

NSErrorDomain const ExecuTorchDumpErrorDomain =
    @"org.pytorch.executorch.etdump";

static NSError *DumpError(ExecuTorchDumpErrorCode code, NSString *message) {
  return [NSError errorWithDomain:ExecuTorchDumpErrorDomain
                             code:code
                         userInfo:@{NSLocalizedDescriptionKey : message}];
}

// A concrete event tracer backed by an ETDumpGen. The base class owns the
// generator through its C++ tracer; this subclass keeps a typed pointer to the
// same object so the recorder can read the trace back without a downcast. The
// pointer stays valid for as long as the module the tracer was given to lives.
@interface ExecuTorchDumpTracer : ExecuTorchEventTracer
- (instancetype)init;
@property(nonatomic, readonly) ETDumpGen *generator;
@end

@implementation ExecuTorchDumpTracer

- (instancetype)init {
  auto generator = std::make_unique<ETDumpGen>();
  _generator = generator.get();
  return [super initWithCppTracer:std::move(generator)];
}

@end

@implementation ExecuTorchDump {
  ExecuTorchModule *_module;
  ExecuTorchDumpTracer *_tracer;
  // Serialises one take against another. Completing a trace finalises the buffer
  // and resets the generator, so two takes must not overlap. This does not guard
  // a take against a concurrent run: the caller must not take while a method is
  // running on another thread, since the generator is being written then.
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
  if (!ExecuTorchDump.isAvailable) {
    if (error) {
      *error = DumpError(
          ExecuTorchDumpErrorCodeUnavailable,
          @"This ExecuTorch runtime was built without event tracing, so no "
          @"profile can be recorded.");
    }
    return nil;
  }
  self = [super init];
  if (self) {
    _lock = [NSLock new];
    _tracer = [[ExecuTorchDumpTracer alloc] init];
    _module = [[ExecuTorchModule alloc] initWithFilePath:filePath
                                           dataFilePaths:dataFilePaths
                                                loadMode:loadMode
                                             eventTracer:_tracer];
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
  ETDumpGen *generator = _tracer.generator;
  // get_etdump_data() finalises the builder and leaves the generator in its
  // Done state, which a second call does not handle and would abort on. Reset
  // after a successful take so a report-on-a-timer caller gets NoData on the
  // next call rather than a process abort.
  ETDumpResult result = generator->get_etdump_data();
  if (result.buf != nullptr && result.size != 0) {
    generator->reset();
  }
  [_lock unlock];
  if (result.buf == nullptr || result.size == 0) {
    if (error) {
      *error = DumpError(ExecuTorchDumpErrorCodeNoData,
                         @"Nothing has been recorded since the last read. "
                         @"Run a method first.");
    }
    return nil;
  }
  // The tracer allocated this buffer with flatcc's aligned allocation and hands
  // it over, so copy it into an object with Cocoa's lifetime and free it with
  // the matching flatcc deallocator on every path out.
  NSData *data = [NSData dataWithBytes:result.buf length:result.size];
  flatcc_builder_aligned_free(result.buf);
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
