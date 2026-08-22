//
// MLMultiArray+Copy.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <CoreML/CoreML.h>

NS_ASSUME_NONNULL_BEGIN

@interface MLMultiArray (Copy)

/// Copies this into another `MLMutiArray`.
///
/// @param dstMultiArray The destination `MLMutiArray`.
- (void)copyInto:(MLMultiArray*)dstMultiArray;

@end

NS_ASSUME_NONNULL_END
