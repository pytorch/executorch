//
// ETCoreMLModelExecutor.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <CoreML/CoreML.h>

@class ETCoreMLModel;

namespace executorchcoreml {
struct ModelLoggingOptions;
class ModelEventLogger;
}

NS_ASSUME_NONNULL_BEGIN
/// A protocol that an object must adopt for hooking into `ETCoreMLModelManager` as a model executor.
@protocol ETCoreMLModelExecutor <NSObject>

/// Executes the model with the given inputs and returns the outputs.
///
/// @param inputs The inputs to the model.
/// @param predictionOptions The prediction options.
/// @param loggingOptions The logging options.
/// @param eventLogger The event logger.
/// @param error   On failure, error is filled with the failure information.
- (nullable NSArray<MLMultiArray*>*)executeModelWithInputs:(id<MLFeatureProvider>)inputs
                                         predictionOptions:(MLPredictionOptions*)predictionOptions
                                            loggingOptions:(const executorchcoreml::ModelLoggingOptions&)loggingOptions
                                               eventLogger:
                                                   (const executorchcoreml::ModelEventLogger* _Nullable)eventLogger
                                                     error:(NSError* __autoreleasing*)error;

/// The model.
@property (readonly, strong, nonatomic) ETCoreMLModel* model;

/// If set to `YES` then output backing are ignored.
@property (readwrite, atomic) BOOL ignoreOutputBackings;


@end

NS_ASSUME_NONNULL_END
