//
// ETCoreMLModelManager.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <CoreML/CoreML.h>

NS_ASSUME_NONNULL_BEGIN

namespace executorchcoreml {
struct ModelLoggingOptions;
class ModelEventLogger;
};

@class ETCoreMLModel;
@class ETCoreMLAssetManager;

typedef void ModelHandle;

/// A class responsible for managing the models loaded by the delegate.
__attribute__((objc_subclassing_restricted))
@interface ETCoreMLModelManager : NSObject

+ (instancetype)new NS_UNAVAILABLE;

- (instancetype)init NS_UNAVAILABLE;

/// Constructs an `ETCoreMLModelManager` instance.
///
/// @param assetManager The asset manager used to manage storage of compiled models.
- (instancetype)initWithAssetManager:(ETCoreMLAssetManager*)assetManager NS_DESIGNATED_INITIALIZER;

/// Loads the model from the AOT  data.
///
/// The data is the AOT blob stored in the executorch Program. The method first parses the model
/// metadata stored in the blob and extracts the identifier. If the asset store contains an asset
/// with the same identifier then the model is directly loaded from the asset otherwise the model is
/// saved to the filesystem, compiled, moved to the assets store, and then loaded from there.
///
/// @param data The AOT blob data.
/// @param configuration The model configuration that will be used to load the model.
/// @param error   On failure, error is filled with the failure information.
/// @retval An opaque handle that points to the loaded model.
- (ModelHandle*)loadModelFromAOTData:(NSData*)data
                       configuration:(MLModelConfiguration*)configuration
                               error:(NSError* __autoreleasing*)error;

/// Executes the loaded model.
///
/// @param handle The handle to the loaded model.
/// @param args The arguments to the model.
/// @param loggingOptions The model logging options.
/// @param error   On failure, error is filled with the failure information.
/// @retval `YES` if the execution succeeded otherwise `NO`.
- (BOOL)executeModelWithHandle:(ModelHandle*)handle
                          args:(NSArray<MLMultiArray*>*)args
                loggingOptions:(const executorchcoreml::ModelLoggingOptions&)loggingOptions
                   eventLogger:(const executorchcoreml::ModelEventLogger* _Nullable)eventLogger
                         error:(NSError* __autoreleasing*)error;

/// Unloads the loaded model.
///
/// @param handle The handle to the loaded model.
/// @retval `YES` if the model is unloaded otherwise `NO`.
- (BOOL)unloadModelWithHandle:(ModelHandle*)handle;

/// Returns the loaded model associated with the handle.
///
/// @param handle The handle to the loaded model.
/// @retval The loaded model.
- (nullable ETCoreMLModel*)modelWithHandle:(ModelHandle*)handle;

/// Pre-warms most recently used assets. This does an advisory read ahead of the asset (compiled
/// model) files and could potentially improve the read time if the data is already available in the
/// system cache.
///
/// @param maxCount The maximum count of assets to be pre-warmed.
- (void)prewarmRecentlyUsedAssetsWithMaxCount:(NSUInteger)maxCount;

/// Pre-warms the model associated with the handle. This could potentially improve the model
/// execution time.
///
/// @param handle The handle to the loaded model.
/// @param error   On failure, error is filled with the failure information.
/// @retval `YES` if the model was pre-warmed otherwise `NO`.
- (BOOL)prewarmModelWithHandle:(ModelHandle*)handle error:(NSError* __autoreleasing*)error;

/// The `ETCoreMLAssetManager` instance used to manage models cache.
@property (strong, readonly, nonatomic) ETCoreMLAssetManager* assetManager;

@end

NS_ASSUME_NONNULL_END
