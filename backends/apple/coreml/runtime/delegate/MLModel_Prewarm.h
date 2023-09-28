//
// MLModel+Prewarm.h
//
// Copyright © 2023 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.


#import <CoreML/CoreML.h>


NS_ASSUME_NONNULL_BEGIN

@interface MLModel (Prewarm)

/// Pre-warms the model by running a prediction with zeroed-out inputs.
///
/// @param error   On failure, error is filled with the failure information.
/// @retval `YES` if the prediction succeeded otherwise `NO`.
- (BOOL)prewarmAndReturnError:(NSError* __autoreleasing*)error;

@end

NS_ASSUME_NONNULL_END
