/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Foundation/Foundation.h>

#import "ExecuTorchBackendOption.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * An immutable, opaque container for per-delegate load-time configuration,
 * built from a dictionary mapping backend identifiers to arrays of
 * `ExecuTorchBackendOption` objects.
 *
 * # Lifetime
 *
 * Once a `BackendOptionsMap` is passed to a `Module` load call, the `Module`
 * **retains** it for as long as the underlying program references it. The
 * caller does not need to manage this lifetime manually — ARC handles it.
 *
 * # Reuse
 *
 * The same `BackendOptionsMap` instance can be reused across multiple `Module`s
 * and across multiple load calls. Build it once, pass it many times.
 *
 * # Validation
 *
 * Validation (option-key length, string-value length, integer 32-bit range)
 * happens at construction time. If the input dictionary contains an invalid
 * entry, the initializer returns `nil` and populates the out-error.
 *
 * @note The current C++ runtime stores integer option values as 32-bit `int`.
 *       Passing an integer outside `[INT32_MIN, INT32_MAX]` will cause the
 *       initializer to fail with `Error::InvalidArgument`.
 */
NS_SWIFT_NAME(BackendOptionsMap)
__attribute__((objc_subclassing_restricted))
@interface ExecuTorchBackendOptionsMap : NSObject

/**
 * Creates a backend options map from a dictionary of per-backend options.
 *
 * @param options A dictionary mapping backend identifiers (e.g. "CoreMLBackend")
 *        to arrays of `ExecuTorchBackendOption` objects configuring that backend.
 * @param error  On failure, populated with an `NSError` describing the validation
 *        problem (e.g. invalid integer range).
 * @return A new instance, or `nil` if validation fails.
 */
- (nullable instancetype)initWithOptions:(NSDictionary<NSString *, NSArray<ExecuTorchBackendOption *> *> *)options
                                   error:(NSError **)error
    NS_DESIGNATED_INITIALIZER NS_SWIFT_NAME(init(options:));

/**
 * Convenience class factory mirroring `-initWithOptions:error:`.
 */
+ (nullable instancetype)mapWithOptions:(NSDictionary<NSString *, NSArray<ExecuTorchBackendOption *> *> *)options
                                  error:(NSError **)error
    NS_RETURNS_RETAINED;

/**
 * The options the receiver was constructed with, exposed as a deep-immutable
 * snapshot dictionary captured at construction time. Useful for debugging and
 * round-tripping.
 */
@property (nonatomic, readonly) NSDictionary<NSString *, NSArray<ExecuTorchBackendOption *> *> *options;

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
