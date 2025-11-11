//
// ETCoreMLModelDebugger.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <CoreML/CoreML.h>

@class ETCoreMLAsset;
@class ETCoreMLAssetManager;
@class ETCoreMLModelDebugInfo;
@class ETCoreMLModelStructurePath;

typedef NSDictionary<ETCoreMLModelStructurePath*, MLMultiArray*> ETCoreMLModelOutputs;

NS_ASSUME_NONNULL_BEGIN
/// A class responsible for debugging a model.
__attribute__((objc_subclassing_restricted))
@interface ETCoreMLModelDebugger : NSObject

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

/// Constructs an `ETCoreMLModelDebugger` instance.
///
/// @param modelAsset The model asset (mlpackage).
/// @param modelDebugInfo The model debug info.
/// @param outputNames The model output names.
/// @param configuration The model configuration.
/// @param assetManager The asset manager used to manage storage of compiled models.
/// @param error   On failure, error is filled with the failure information.
- (nullable instancetype)initWithModelAsset:(ETCoreMLAsset*)modelAsset
                             modelDebugInfo:(nullable ETCoreMLModelDebugInfo*)modelDebugInfo
                                outputNames:(NSOrderedSet<NSString*>*)outputNames
                              configuration:(MLModelConfiguration*)configuration
                               assetManager:(ETCoreMLAssetManager*)assetManager
                                      error:(NSError* __autoreleasing*)error NS_DESIGNATED_INITIALIZER;

/// Returns outputs of operations at the specified paths.
///
/// This is an expensive method, it creates models with intermediate outputs, compiles, and executes them.
///
/// @param paths The operation paths.
/// @param options The prediction options.
/// @param inputs The model inputs..
/// @param modelOutputs  On success, modelOutputs is filled with the model outputs.
/// @param error   On failure, error is filled with the failure information.
/// @retval A dictionary with the operation path as the key and the operation output as the value.
- (nullable ETCoreMLModelOutputs*)
    outputsOfOperationsAtPaths:(NSArray<ETCoreMLModelStructurePath*>*)paths
                       options:(MLPredictionOptions*)options
                        inputs:(id<MLFeatureProvider>)inputs
                  modelOutputs:(NSArray<MLMultiArray*>* _Nullable __autoreleasing* _Nonnull)modelOutputs
                         error:(NSError* __autoreleasing*)error;

/// The paths to all the operations for which we can get the outputs.
@property (readonly, copy, nonatomic) NSArray<ETCoreMLModelStructurePath*>* operationPaths;

/// Operation path to debug symbol map.
@property (readonly, copy, nonatomic)
    NSDictionary<ETCoreMLModelStructurePath*, NSString*>* operationPathToDebugSymbolMap;

@end

NS_ASSUME_NONNULL_END
