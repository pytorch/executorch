/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <ExecuTorch/ExecuTorchEventTracer.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * An event tracer that records an ETDump trace.
 *
 * Pass one to a `Module` at creation to have that module record through it, then
 * read the trace back with `Dump`. `Dump` uses this internally; construct one
 * directly only to attach ETDump recording to a `Module` you build yourself.
 */
NS_SWIFT_NAME(DumpTracer)
@interface ExecuTorchDumpTracer : ExecuTorchEventTracer

- (instancetype)init;

/**
 * Takes the trace recorded so far, in the ETDump format.
 *
 * Completing the trace ends it, so this returns each recorded span once and a
 * subsequent call reports that there is nothing new. Run a method on the module
 * this tracer was given to first.
 *
 * @param error On failure, describes why, most often that nothing has run yet.
 * @return The trace, or nil.
 */
- (nullable NSData *)takeDataWithError:(NSError **)error
    NS_SWIFT_NAME(takeData());

/**
 * Takes the trace recorded so far and writes it to a file.
 *
 * @param path The file to write. An existing file is replaced.
 * @param error On failure, describes why.
 * @return YES if a trace was written.
 */
- (BOOL)takeDataToFile:(NSString *)path
                 error:(NSError **)error NS_SWIFT_NAME(takeData(toFile:));

@end

NS_ASSUME_NONNULL_END
