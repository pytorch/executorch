//
// ETCoreMLModel.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <CoreML/CoreML.h>
#import <vector>

NS_ASSUME_NONNULL_BEGIN

@class ETCoreMLAsset;

namespace executorchcoreml {
class MultiArray;
}

/// Represents a ML model, the class is a thin wrapper over `MLModel` with additional properties.
__attribute__((objc_subclassing_restricted))
@interface ETCoreMLModel : NSObject

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

/// Constructs an `ETCoreMLModel` instance.
///
/// @param asset The asset from which the model is loaded.
/// @param configuration The model configuration.
/// @param orderedInputNames   The ordered input names of the model.
/// @param orderedOutputNames   The ordered output names of the model.
/// @param error   On failure, error is filled with the failure information.
- (nullable instancetype)initWithAsset:(ETCoreMLAsset*)asset
                         configuration:(MLModelConfiguration*)configuration
                     orderedInputNames:(NSOrderedSet<NSString*>*)orderedInputNames
                    orderedOutputNames:(NSOrderedSet<NSString*>*)orderedOutputNames
                                 error:(NSError* __autoreleasing*)error NS_DESIGNATED_INITIALIZER;

/// The underlying MLModel.
@property (strong, readonly, nonatomic) MLModel* mlModel;

/// The model state.
@property (strong, readonly, nonatomic, nullable) id state;

/// The asset from which the model is loaded.
@property (strong, readonly, nonatomic) ETCoreMLAsset* asset;

/// The asset identifier.
@property (strong, readonly, nonatomic) NSString* identifier;

/// The ordered input names of the model.
@property (copy, readonly, nonatomic) NSOrderedSet<NSString*>* orderedInputNames;

/// The ordered output names of the model.
@property (copy, readonly, nonatomic) NSOrderedSet<NSString*>* orderedOutputNames;


- (nullable id<MLFeatureProvider>)predictionFromFeatures:(id<MLFeatureProvider>)input
                                                 options:(MLPredictionOptions*)options
                                                   error:(NSError* __autoreleasing*)error;

- (nullable NSArray<MLMultiArray*>*)prepareInputs:(const std::vector<executorchcoreml::MultiArray>&)inputs
                                            error:(NSError* __autoreleasing*)error;

- (nullable NSArray<MLMultiArray*>*)prepareOutputBackings:(const std::vector<executorchcoreml::MultiArray>&)outputs
                                                    error:(NSError* __autoreleasing*)error;

- (BOOL)prewarmAndReturnError:(NSError* __autoreleasing*)error;

@end

NS_ASSUME_NONNULL_END
