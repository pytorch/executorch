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

/**
 * Initializes a tracer with a native std::unique_ptr<EventTracer> instance.
 *
 * @param nativeInstance A pointer to a native std::unique_ptr<EventTracer>
 * instance.
 * @return An initialized ExecuTorchEventTracer instance.
 */
- (instancetype)initWithNativeInstance:(void *)nativeInstance
    NS_DESIGNATED_INITIALIZER NS_SWIFT_UNAVAILABLE("");

/**
 * Pointer to the underlying native std::unique_ptr<EventTracer> instance.
 *
 * @return A raw pointer to the native std::unique_ptr<EventTracer> held by this
 * class.
 */
@property(nonatomic, readonly) void *nativeInstance NS_SWIFT_UNAVAILABLE("");

@end

NS_ASSUME_NONNULL_END
