//
// ETCoreMLPair.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN
/// A class representing a pair with first and second objects.
__attribute__((objc_subclassing_restricted))
@interface ETCoreMLPair<First, Second> : NSObject<NSCopying>

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

/// Constructs an `ETCoreMLPair` instance.
///
/// @param first The first object of this pair.
/// @param second The second object of this pair.
- (instancetype)initWithFirst:(First)first second:(Second)second NS_DESIGNATED_INITIALIZER;

/// The first object.
@property (nonatomic, readonly) First first;

/// The second object..
@property (nonatomic, readonly) Second second;

@end

NS_ASSUME_NONNULL_END
