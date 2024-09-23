//
// ETCoreMLModelAnalyzer.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <CoreML/CoreML.h>

#import "ETCoreMLModelExecutor.h"

namespace executorchcoreml {
struct ModelMetadata;
}

@class ETCoreMLAsset;
@class ETCoreMLAssetManager;
@class ETCoreMLModelDebugInfo;
@class ETCoreMLModelStructurePath;
@protocol ETCoreMLModelEventLogger;

NS_ASSUME_NONNULL_BEGIN

__attribute__((objc_subclassing_restricted))
/// A class responsible for executing a model, it also logs model events (profiling and debugging ).
@interface ETCoreMLModelAnalyzer : NSObject<ETCoreMLModelExecutor>

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

/// Constructs an `ETCoreMLModelAnalyzer` instance.
///
/// @param compiledModelAsset The compiled model asset (mlmodelc).
/// @param modelAsset The model asset (mlpackage).
/// @param modelDebugInfo The model debug info.
/// @param metadata The model metadata.
/// @param configuration The model configuration.
/// @param assetManager The asset manager used to manage storage of compiled models.
/// @param error   On failure, error is filled with the failure information.
- (nullable instancetype)initWithCompiledModelAsset:(ETCoreMLAsset*)compiledModelAsset
                                         modelAsset:(nullable ETCoreMLAsset*)modelAsset
                                     modelDebugInfo:(nullable ETCoreMLModelDebugInfo*)modelDebugInfo
                                           metadata:(const executorchcoreml::ModelMetadata&)metadata
                                      configuration:(MLModelConfiguration*)configuration
                                       assetManager:(ETCoreMLAssetManager*)assetManager
                                              error:(NSError* __autoreleasing*)error NS_DESIGNATED_INITIALIZER;
/// The model.
@property (readonly, strong, nonatomic) ETCoreMLModel* model;

/// If set to `YES` then output backing are ignored.
@property (readwrite, atomic) BOOL ignoreOutputBackings;

@end

NS_ASSUME_NONNULL_END
