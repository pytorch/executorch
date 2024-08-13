//
// ETCoreMLModelCompiler.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <CoreML/CoreML.h>

NS_ASSUME_NONNULL_BEGIN
/// A class responsible for compiling a CoreML model.
__attribute__((objc_subclassing_restricted))
@interface ETCoreMLModelCompiler : NSObject

+ (instancetype)new NS_UNAVAILABLE;

- (instancetype)init NS_UNAVAILABLE;

/// Synchronously compiles a model given the location of its on-disk representation.
///
/// @param modelURL The location of the model's on-disk representation (.mlpackage directory).
/// @param maxWaitTimeInSeconds The maximum wait time in seconds.
/// @param error   On failure, error is filled with the failure information.
+ (nullable NSURL*)compileModelAtURL:(NSURL*)modelURL
                maxWaitTimeInSeconds:(NSTimeInterval)maxWaitTimeInSeconds
                               error:(NSError* __autoreleasing*)error;

@end

NS_ASSUME_NONNULL_END
