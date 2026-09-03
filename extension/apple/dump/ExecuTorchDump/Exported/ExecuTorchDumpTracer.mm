/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchDumpTracer.h"

#import "ExecuTorchDumpError.h"
#import "ExecuTorchDumpTracer+Internal.h"

#import <flatcc/flatcc_builder.h>

using namespace executorch::etdump;

static NSError *DumpError(ExecuTorchDumpErrorCode code, NSString *message) {
  return [NSError errorWithDomain:ExecuTorchDumpErrorDomain
                             code:code
                         userInfo:@{NSLocalizedDescriptionKey : message}];
}

@implementation ExecuTorchDumpTracer {
  // A typed pointer to the generator the base class owns through its C++ tracer,
  // so the recorder can read the trace back without a downcast. It stays valid
  // for as long as the module the tracer was given to lives.
  ETDumpGen *_generator;
  // Holds a trace that was extracted but not yet handed to the caller, so a write
  // that fails does not lose it: extracting finalises and resets the generator and
  // cannot be undone, so the bytes are kept here and returned on the next take.
  NSData *_pendingData;
  // Serialises one take against another. Completing a trace finalises the buffer
  // and resets the generator, so two takes must not overlap. It does not guard a
  // take against a concurrent run: the caller must not take while a method is
  // running on another thread, since the generator is being written then.
  NSLock *_lock;
}

- (instancetype)init {
  auto generator = std::make_unique<ETDumpGen>();
  _generator = generator.get();
  // The base moves out of the pointee as a std::unique_ptr<EventTracer>, so the
  // local must be that type, not unique_ptr<ETDumpGen>. _generator keeps the
  // concrete pointer for readback.
  std::unique_ptr<executorch::runtime::EventTracer> tracer =
      std::move(generator);
  self = [super initWithNativeInstance:&tracer];
  if (self) {
    _lock = [NSLock new];
  }
  return self;
}

- (ETDumpGen *)generator {
  return _generator;
}

- (nullable NSData *)takeDataWithError:(NSError **)error {
  [_lock lock];
  // A previous takeDataToFile: may have extracted a trace and then failed to
  // write it. Hand that back rather than reporting nothing recorded.
  if (_pendingData != nil) {
    NSData *pending = _pendingData;
    _pendingData = nil;
    [_lock unlock];
    return pending;
  }
  // get_etdump_data() finalises the builder and leaves the generator in its Done
  // state, which a second call does not handle and would abort on. Reset after a
  // successful take so a report-on-a-timer caller gets NoData on the next call
  // rather than a process abort.
  ETDumpResult result = _generator->get_etdump_data();
  if (result.buf != nullptr && result.size != 0) {
    _generator->reset();
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
  // it over, so copy it into an object with Cocoa's lifetime and free it with the
  // matching flatcc deallocator on every path out.
  NSData *data = [NSData dataWithBytes:result.buf length:result.size];
  flatcc_builder_aligned_free(result.buf);
  return data;
}

- (BOOL)takeDataToFile:(NSString *)path error:(NSError **)error {
  NSData *data = [self takeDataWithError:error];
  if (data == nil) {
    return NO;
  }
  if (![data writeToFile:path options:NSDataWritingAtomic error:error]) {
    // Extracting the trace finalised and reset the generator, so it is gone from
    // there. Keep it so the caller can retry the write rather than lose it.
    [_lock lock];
    _pendingData = data;
    [_lock unlock];
    return NO;
  }
  return YES;
}

@end
