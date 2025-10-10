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
 A configuration object for text generation.

 This class wraps the underlying C++ GenerationConfig so that default
 values and future fields remain a single source of truth in C++.
*/
NS_SWIFT_NAME(Config)
__attribute__((deprecated("This API is experimental.")))
__attribute__((objc_subclassing_restricted))
@interface ExecuTorchLLMConfig : NSObject<NSCopying>

/** Whether to echo the input prompt in the output. */
@property(nonatomic, getter=isEchoEnabled) BOOL echoEnabled;

/** Maximum number of new tokens to generate. */
@property(nonatomic) NSInteger maximumNewTokens;

/** Whether this is a warmup run. */
@property(nonatomic, getter=isWarming) BOOL warming;

/** Maximum total sequence length. */
@property(nonatomic) NSInteger sequenceLength;

/** Temperature for sampling. */
@property(nonatomic) double temperature;

/** Number of BOS tokens to add. */
@property(nonatomic) NSInteger bosCount;

/** Number of EOS tokens to add. */
@property(nonatomic) NSInteger eosCount;

/**
 Initializes a configuration and invokes the block to mutate it.

 @param block  A block that receives the newly initialized configuration.
 @return An initialized ExecuTorchLLMConfig instance.
*/
- (instancetype)initWithBlock:(NS_NOESCAPE void (^)(ExecuTorchLLMConfig *))block
    NS_SWIFT_NAME(init(_:));

@end

NS_ASSUME_NONNULL_END
