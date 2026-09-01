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
 * An opaque handle to a runtime event tracer.
 *
 * A tracer records what a model does while it runs. This base class carries no
 * behavior of its own: a concrete tracer, such as the one the ExecuTorchDump
 * framework provides, subclasses it and supplies the underlying implementation.
 * Pass an instance to a `Module` at creation to have that module record through
 * it.
 *
 * # Lifetime
 * A tracer is single-use: creating a `Module` with it hands the underlying
 * implementation to that module, which owns it from then on. Do not pass the
 * same instance to a second module.
 */
NS_SWIFT_NAME(EventTracer)
@interface ExecuTorchEventTracer : NSObject

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
