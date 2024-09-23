//
// ETCoreMLModelProfiler.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "ETCoreMLPair.h"
#import <CoreML/CoreML.h>
#import <TargetConditionals.h>

@class ETCoreMLAsset;
@class ETCoreMLModel;
@class ETCoreMLModelStructurePath;
@class ETCoreMLOperationProfilingInfo;

typedef NSDictionary<ETCoreMLModelStructurePath*, ETCoreMLOperationProfilingInfo*> ETCoreMLModelProfilingResult;

#if !defined(MODEL_PROFILING_IS_AVAILABLE) && __has_include(<CoreML/MLComputePlan.h>)
#define MODEL_PROFILING_IS_AVAILABLE 1
#endif

NS_ASSUME_NONNULL_BEGIN
/// A class responsible for profiling a model.
__attribute__((objc_subclassing_restricted))
@interface ETCoreMLModelProfiler : NSObject

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

/// Constructs an `ETCoreMLModelProfiler` instance.
///
/// @param model The model.
/// @param configuration The model configuration.
/// @param error   On failure, error is filled with the failure information.
- (nullable instancetype)initWithModel:(ETCoreMLModel*)model
                         configuration:(MLModelConfiguration*)configuration
                                 error:(NSError* __autoreleasing*)error NS_DESIGNATED_INITIALIZER;

/// Returns profiling info of operations at the specified paths.
///
/// @param paths The operation paths.
/// @param options The prediction options.
/// @param inputs The model inputs..
/// @param modelOutputs  On success, modelOutputs is filled with the model outputs.
/// @param error   On failure, error is filled with the failure information.
/// @retval A dictionary with the operation path as the key and the profiling info as the value.
- (nullable ETCoreMLModelProfilingResult*)
    profilingInfoForOperationsAtPaths:(NSArray<ETCoreMLModelStructurePath*>*)paths
                              options:(MLPredictionOptions*)options
                               inputs:(id<MLFeatureProvider>)inputs
                         modelOutputs:(NSArray<MLMultiArray*>* _Nullable __autoreleasing* _Nonnull)modelOutputs
                                error:(NSError* __autoreleasing*)error;

/// The paths to all the operations for which we can get the profiling info.
@property (readonly, copy, nonatomic) NSArray<ETCoreMLModelStructurePath*>* operationPaths;

@end

NS_ASSUME_NONNULL_END
