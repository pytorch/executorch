//
// MLModel+Prewarm.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.


#import <CoreML/CoreML.h>

#if !defined(MODEL_STATE_IS_SUPPORTED) && __has_include(<CoreML/MLModel+MLState.h>)
#define MODEL_STATE_IS_SUPPORTED 1
#endif

NS_ASSUME_NONNULL_BEGIN

@interface MLModel (Prewarm)

/// Pre-warms the model by running a prediction with zeroed-out inputs.
///
/// @param state The model state.
/// @param error   On failure, error is filled with the failure information.
/// @retval `YES` if the prediction succeeded otherwise `NO`.
- (BOOL)prewarmUsingState:(nullable id)state error:(NSError* __autoreleasing*)error;

@end

NS_ASSUME_NONNULL_END
