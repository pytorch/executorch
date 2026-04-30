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
 * Enum to define the type of a backend option value.
 */
typedef NS_ENUM(NSInteger, ExecuTorchBackendOptionType) {
  ExecuTorchBackendOptionTypeBoolean,
  ExecuTorchBackendOptionTypeInteger,
  ExecuTorchBackendOptionTypeString,
} NS_SWIFT_NAME(BackendOptionType);

/**
 * Represents a single key-value configuration option for a backend.
 *
 * Backend options are used to pass per-delegate configuration (e.g., compute
 * unit, thread count, cache directory) when loading a module. Each option has
 * a string key and a typed value (boolean, integer, or string).
 */
NS_SWIFT_NAME(BackendOption)
__attribute__((objc_subclassing_restricted))
@interface ExecuTorchBackendOption : NSObject

/** The option key name (e.g. "compute_unit", "num_threads"). */
@property (nonatomic, readonly) NSString *key;

/** The type of the option value. */
@property (nonatomic, readonly) ExecuTorchBackendOptionType type;

/** The boolean value. Only valid when type is Boolean. */
@property (nonatomic, readonly) BOOL boolValue;

/** The integer value. Only valid when type is Integer. */
@property (nonatomic, readonly) NSInteger intValue;

/** The string value. Only valid when type is String. */
@property (nullable, nonatomic, readonly) NSString *stringValue;

/**
 * Creates a backend option with a boolean value.
 *
 * @param key The option key.
 * @param value The boolean value.
 * @return A new ExecuTorchBackendOption instance.
 */
+ (instancetype)optionWithKey:(NSString *)key
                 booleanValue:(BOOL)value
    NS_SWIFT_NAME(init(_:_:))
    NS_RETURNS_RETAINED;

/**
 * Creates a backend option with an integer value.
 *
 * @param key The option key.
 * @param value The integer value.
 * @return A new ExecuTorchBackendOption instance.
 */
+ (instancetype)optionWithKey:(NSString *)key
                 integerValue:(NSInteger)value
    NS_SWIFT_NAME(init(_:_:))
    NS_RETURNS_RETAINED;

/**
 * Creates a backend option with a string value.
 *
 * @param key The option key.
 * @param value The string value.
 * @return A new ExecuTorchBackendOption instance.
 */
+ (instancetype)optionWithKey:(NSString *)key
                  stringValue:(NSString *)value
    NS_SWIFT_NAME(init(_:_:))
    NS_RETURNS_RETAINED;

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
