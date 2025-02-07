//
// ETCoreMLModelAnalyzer.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "ETCoreMLModelAnalyzer.h"

#import "ETCoreMLAsset.h"
#import "ETCoreMLAssetManager.h"
#import "ETCoreMLDefaultModelExecutor.h"
#import "ETCoreMLLogging.h"
#import "ETCoreMLModel.h"
#import "ETCoreMLModelLoader.h"
#import "ETCoreMLModelStructurePath.h"
#import "ETCoreMLModelDebugInfo.h"
#import "ETCoreMLModelDebugger.h"
#import "ETCoreMLModelProfiler.h"
#import "ETCoreMLStrings.h"
#import "model_logging_options.h"
#import "model_event_logger.h"
#import "model_metadata.h"
#import "model_package_info.h"
#import "objc_safe_cast.h"

namespace {
using namespace executorchcoreml;
static constexpr NSInteger MAX_MODEL_OUTPUTS_COUNT = 50;
} //namespace

@interface ETCoreMLModelAnalyzer ()

@property (readonly, strong, nonatomic) ETCoreMLAsset *modelAsset;
@property (readonly, strong, nonatomic) ETCoreMLAssetManager *assetManager;
@property (strong, nonatomic, nullable) ETCoreMLModelProfiler *profiler;
@property (strong, nonatomic, nullable) ETCoreMLModelDebugger *debugger;
@property (strong, nonatomic, nullable) id<ETCoreMLModelExecutor> executor;
@property (readonly, copy, nonatomic, nullable) ETCoreMLModelDebugInfo *modelDebugInfo;
@property (readonly, strong, nonatomic) MLModelConfiguration *configuration;

@end

@implementation ETCoreMLModelAnalyzer

- (nullable instancetype)initWithCompiledModelAsset:(ETCoreMLAsset *)compiledModelAsset
                                         modelAsset:(nullable ETCoreMLAsset *)modelAsset
                                     modelDebugInfo:(nullable ETCoreMLModelDebugInfo *)modelDebugInfo
                                           metadata:(const executorchcoreml::ModelMetadata&)metadata
                                      configuration:(MLModelConfiguration *)configuration
                                       assetManager:(ETCoreMLAssetManager *)assetManager
                                              error:(NSError * __autoreleasing *)error {
    if (modelAsset && ![modelAsset keepAliveAndReturnError:error]) {
        return nil;
    }
    
    if (![compiledModelAsset keepAliveAndReturnError:error]) {
        return nil;
    }
    
    NSError *localError = nil;
    ETCoreMLModel *model = [ETCoreMLModelLoader loadModelWithContentsOfURL:compiledModelAsset.contentURL
                                                             configuration:configuration
                                                                  metadata:metadata
                                                              assetManager:assetManager
                                                                     error:&localError];
    if (!model) {
        ETCoreMLLogError(localError,
                         "%@: Failed to create model profiler.",
                         NSStringFromClass(ETCoreMLAssetManager.class));
    }
    
    self = [super init];
    if (self) {
        _model = model;
        _modelAsset = modelAsset;
        _modelDebugInfo = modelDebugInfo;
        _assetManager = assetManager;
        _configuration = configuration;
        _executor = [[ETCoreMLDefaultModelExecutor alloc] initWithModel:model];
    }
    
    return self;
}

- (nullable NSArray<MLMultiArray *> *)profileModelWithInputs:(id<MLFeatureProvider>)inputs
                                           predictionOptions:(MLPredictionOptions *)predictionOptions
                                                 eventLogger:(const executorchcoreml::ModelEventLogger *)eventLogger
                                                       error:(NSError * __autoreleasing *)error {
    if (self.profiler == nil) {
        ETCoreMLModelProfiler *profiler = [[ETCoreMLModelProfiler alloc] initWithModel:self.model
                                                                         configuration:self.configuration
                                                                                 error:error];
        self.profiler = profiler;
    }
       
    
    if (!self.profiler) {
        ETCoreMLLogErrorAndSetNSError(error,
                                      ETCoreMLErrorModelProfilingNotSupported,
                                      "%@: Model profiling is only available for macOS >= 14.4, iOS >= 17.4, tvOS >= 17.4 and watchOS >= 10.4.",
                                      NSStringFromClass(ETCoreMLModelAnalyzer.class));
        return nil;
    }
    
    NSArray<MLMultiArray *> *modelOutputs = nil;
    NSArray<ETCoreMLModelStructurePath *> *operationPaths = self.profiler.operationPaths;
    ETCoreMLModelProfilingResult *profilingInfos = [self.profiler profilingInfoForOperationsAtPaths:operationPaths
                                                                                            options:predictionOptions
                                                                                             inputs:inputs
                                                                                       modelOutputs:&modelOutputs
                                                                                              error:error];
    if (!profilingInfos) {
        return nil;
    }
    
    eventLogger->log_profiling_infos(profilingInfos, self.modelDebugInfo.pathToDebugSymbolMap);
    return modelOutputs;
}

- (nullable NSArray<MLMultiArray *> *)debugModelWithInputs:(id<MLFeatureProvider>)inputs
                                         predictionOptions:(MLPredictionOptions *)predictionOptions
                                               eventLogger:(const executorchcoreml::ModelEventLogger *)eventLogger
                                                     error:(NSError * __autoreleasing *)error {
    if (!self.modelAsset) {
        ETCoreMLLogErrorAndSetNSError(error,
                                      ETCoreMLErrorCorruptedData,
                                      "%@: There is no mlpackage, mlpackage is required for debugging a model. Please check the export path.",
                                      NSStringFromClass(ETCoreMLModelAnalyzer.class));
        return nil;
    }
    
    if (!self.debugger) {
        self.debugger = [[ETCoreMLModelDebugger alloc] initWithModelAsset:self.modelAsset
                                                           modelDebugInfo:self.modelDebugInfo
                                                              outputNames:self.model.orderedOutputNames
                                                            configuration:self.configuration
                                                             assetManager:self.assetManager
                                                                    error:error];
    }
    
    if (!self.debugger) {
        return nil;
    }
    
    NSArray<MLMultiArray *> *modelOutputs = nil;
    NSArray<ETCoreMLModelStructurePath *> *operationPaths = self.debugger.operationPaths;
    NSDictionary<ETCoreMLModelStructurePath *, NSString *> *operationPathToDebugSymbolMap = self.debugger.operationPathToDebugSymbolMap;
    NSInteger n = operationPaths.count/MAX_MODEL_OUTPUTS_COUNT + (operationPaths.count % MAX_MODEL_OUTPUTS_COUNT == 0 ? 0 : 1);
    for (NSInteger i = 0; i < n; i++) {
        @autoreleasepool {
            NSRange range = NSMakeRange(i * MAX_MODEL_OUTPUTS_COUNT, MIN(operationPaths.count - i * MAX_MODEL_OUTPUTS_COUNT, MAX_MODEL_OUTPUTS_COUNT));
            ETCoreMLModelOutputs *outputs = [self.debugger outputsOfOperationsAtPaths:[operationPaths subarrayWithRange:range]
                                                                              options:predictionOptions
                                                                               inputs:inputs
                                                                         modelOutputs:&modelOutputs
                                                                                error:error];
            if (!outputs) {
                return nil;
            }
            
            if (outputs.count > 0) {
                eventLogger->log_intermediate_tensors(outputs, operationPathToDebugSymbolMap);
            }
        }
    }
    
    return modelOutputs;
}

- (nullable NSArray<MLMultiArray *> *)executeModelWithInputs:(id<MLFeatureProvider>)inputs
                                           predictionOptions:(MLPredictionOptions *)predictionOptions
                                              loggingOptions:(const executorchcoreml::ModelLoggingOptions&)loggingOptions
                                                 eventLogger:(const executorchcoreml::ModelEventLogger* _Nullable)eventLogger
                                                       error:(NSError * __autoreleasing *)error {
    if (self.ignoreOutputBackings) {
        predictionOptions.outputBackings = @{};
    }
    
    NSError *localError = nil;
    NSArray<MLMultiArray *> *outputs = nil;
    if (loggingOptions.log_profiling_info) {
        NSAssert(eventLogger != nullptr, @"%@: Event logger is set to nullptr.", NSStringFromClass(ETCoreMLModelAnalyzer.class));
        outputs = [self profileModelWithInputs:inputs
                             predictionOptions:predictionOptions
                                   eventLogger:eventLogger
                                         error:&localError];
    }
    
    if (loggingOptions.log_intermediate_tensors) {
        NSAssert(eventLogger != nullptr, @"%@: Event logger is set to nullptr.", NSStringFromClass(ETCoreMLModelAnalyzer.class));
        outputs = [self debugModelWithInputs:inputs
                           predictionOptions:predictionOptions
                                 eventLogger:eventLogger
                                       error:&localError];
    }
    
    if (!loggingOptions.log_profiling_info && !loggingOptions.log_intermediate_tensors) {
        outputs = [self.executor executeModelWithInputs:inputs
                                      predictionOptions:predictionOptions
                                         loggingOptions:executorchcoreml::ModelLoggingOptions()
                                            eventLogger:nullptr
                                                  error:&localError];
    }
    
    if (error) {
        *error = localError;
    }
    
    return outputs;
}

@end
