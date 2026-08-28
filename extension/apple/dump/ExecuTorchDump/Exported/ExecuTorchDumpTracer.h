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

- (instancetype)init NS_DESIGNATED_INITIALIZER;

@end

NS_ASSUME_NONNULL_END
