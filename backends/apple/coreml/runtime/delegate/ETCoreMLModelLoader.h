//
// ETCoreMLModelLoader.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <CoreML/CoreML.h>

@class ETCoreMLModel;
@class ETCoreMLAssetManager;

namespace executorchcoreml {
struct ModelMetadata;
}

NS_ASSUME_NONNULL_BEGIN
/// A class responsible for loading a CoreML model.
__attribute__((objc_subclassing_restricted))
@interface ETCoreMLModelLoader : NSObject

+ (instancetype)new NS_UNAVAILABLE;

- (instancetype)init NS_UNAVAILABLE;

/// Synchronously loads a model given the location of its on-disk representation and configuration.
///
/// @param compiledModelURL The location of the model's on-disk representation (.mlmodelc directory).
/// @param configuration The model configuration.
/// @param metadata   The model metadata.
/// @param assetManager The asset manager used to manage storage of compiled models.
/// @param error   On failure, error is filled with the failure information.
+ (nullable ETCoreMLModel*)loadModelWithContentsOfURL:(NSURL*)compiledModelURL
                                        configuration:(MLModelConfiguration*)configuration
                                             metadata:(const executorchcoreml::ModelMetadata&)metadata
                                         assetManager:(ETCoreMLAssetManager*)assetManager
                                                error:(NSError* __autoreleasing*)error;

@end

NS_ASSUME_NONNULL_END
