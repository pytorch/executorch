//
// ETCoreMLModelAnalyzer.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <ETCoreMLAsset.h>
#import <ETCoreMLAssetManager.h>
#import <ETCoreMLDefaultModelExecutor.h>
#import <ETCoreMLLogging.h>
#import <ETCoreMLModel.h>
#import <ETCoreMLModelAnalyzer.h>
#import <ETCoreMLModelLoader.h>
#import <ETCoreMLModelStructurePath.h>
#import <ETCoreMLModelDebugger.h>
#import <ETCoreMLModelProfiler.h>
#import <ETCoreMLStrings.h>
#import <model_logging_options.h>
#import <model_event_logger.h>
#import <model_metadata.h>
#import <model_package_info.h>
#import <objc_safe_cast.h>

namespace {
using namespace executorchcoreml;
static constexpr NSInteger MAX_MODEL_OUTPUTS_COUNT = 50;

NSDictionary<ETCoreMLModelStructurePath *, NSString *> * _Nullable get_path_to_symbol_name_map(ETCoreMLAsset *model_asset,
                                                                                               NSFileManager *fm,
                                                                                               NSError * __autoreleasing *error) {
    auto package_info = ModelPackageInfo::make(model_asset.contentURL, fm, error);
    if (!package_info) {
        return nil;
    }
    
    const auto& items = package_info.value().items;
    auto debug_symbols_file_name = std::string(ETCoreMLStrings.debugSymbols.UTF8String);
    auto it = std::find_if(items.begin(), items.end(), [&debug_symbols_file_name](const auto& pair) {
        return pair.second.name == debug_symbols_file_name;
    });
    
    if (it == items.end()) {
        return nil;
    }
    
    NSURL *debug_symbols_file_url = [model_asset.contentURL URLByAppendingPathComponent:@(it->second.path.c_str())];
    NSData *data = [NSData dataWithContentsOfURL:debug_symbols_file_url
                                         options:NSDataReadingMapped
                                           error:error];
    if (!data) {
        return nil;
    }
    
    id object = [NSJSONSerialization JSONObjectWithData:data options:(NSJSONReadingOptions)0 error:error];
    if (!object) {
        return nil;
    }
    
    NSDictionary<NSString *, id> *json_dict = SAFE_CAST(object, NSDictionary);
    NSCAssert(json_dict != nil, @"The contents of %s is not a json dictionary.", it->second.path.c_str());
    NSMutableDictionary<ETCoreMLModelStructurePath *, NSString *> *result = [NSMutableDictionary dictionaryWithCapacity:json_dict.count];
    for (NSString *symbol_name in json_dict) {
        NSArray<NSDictionary<NSString *, id> *> *components = SAFE_CAST(json_dict[symbol_name], NSArray);
        NSCAssert(components != nil, @"The path=%@ is invalid.", json_dict[symbol_name]);
        ETCoreMLModelStructurePath *path = [[ETCoreMLModelStructurePath alloc] initWithComponents:json_dict[symbol_name]];
        result[path] = symbol_name;
    }
    
    return result;
}
} //namespace

@interface ETCoreMLModelAnalyzer ()

@property (readonly, strong, nonatomic) ETCoreMLAsset *modelAsset;
@property (readonly, strong, nonatomic) ETCoreMLAssetManager *assetManager;
@property (strong, nonatomic) ETCoreMLModelProfiler *profiler;
@property (strong, nonatomic) ETCoreMLModelDebugger *debugger;
@property (strong, nonatomic) id<ETCoreMLModelExecutor> executor;
@property (readonly, copy, nonatomic, nullable) NSDictionary<ETCoreMLModelStructurePath *, NSString *> *pathToSymbolNameMap;
@property (readonly, strong, nonatomic) MLModelConfiguration *configuration;

@end

@implementation ETCoreMLModelAnalyzer

- (nullable instancetype)initWithCompiledModelAsset:(ETCoreMLAsset *)compiledModelAsset
                                         modelAsset:(ETCoreMLAsset *)modelAsset
                                           metadata:(const executorchcoreml::ModelMetadata&)metadata
                                      configuration:(MLModelConfiguration *)configuration
                                       assetManager:(ETCoreMLAssetManager *)assetManager
                                              error:(NSError * __autoreleasing *)error {
    if (![modelAsset keepAliveAndReturnError:error]) {
        return nil;
    }
    
    if (![compiledModelAsset keepAliveAndReturnError:error]) {
        return nil;
    }
    
    ETCoreMLModel *model = [ETCoreMLModelLoader loadModelWithContentsOfURL:compiledModelAsset.contentURL
                                                             configuration:configuration
                                                                  metadata:metadata
                                                              assetManager:assetManager
                                                                     error:error];
    if (!model) {
        return nil;
    }
    
    NSError *localError = nil;
    NSDictionary<ETCoreMLModelStructurePath *, NSString *> *pathToSymbolNameMap = get_path_to_symbol_name_map(modelAsset,
                                                                                                              assetManager.fileManager,
                                                                                                              &localError);
    
    if (localError) {
        os_log_error(ETCoreMLErrorUtils.loggingChannel , "%@: The model package at path=%@ has invalid or missing debug symbols file.",
                     NSStringFromClass(ETCoreMLModelAnalyzer.class),
                     modelAsset.contentURL.path);
    }
    
    ETCoreMLModelProfiler *profiler = [[ETCoreMLModelProfiler alloc] initWithCompiledModelAsset:model.asset
                                                                                    outputNames:model.orderedOutputNames
                                                                                  configuration:configuration
                                                                                          error:error];
    if (!profiler) {
        return nil;
    }
    
    self = [super init];
    if (self) {
        _model = model;
        _modelAsset = modelAsset;
        _assetManager = assetManager;
        _configuration = configuration;
        _pathToSymbolNameMap = pathToSymbolNameMap;
        _executor = [[ETCoreMLDefaultModelExecutor alloc] initWithModel:model];
        _profiler = profiler;
    }
    
    return self;
}

- (nullable NSArray<MLMultiArray *> *)profileModelWithInputs:(id<MLFeatureProvider>)inputs
                                           predictionOptions:(MLPredictionOptions *)predictionOptions
                                                 eventLogger:(const executorchcoreml::ModelEventLogger *)eventLogger
                                                       error:(NSError * __autoreleasing *)error {
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
    
    eventLogger->log_profiling_infos(profilingInfos, self.pathToSymbolNameMap);
    return modelOutputs;
}

- (nullable NSArray<MLMultiArray *> *)debugModelWithInputs:(id<MLFeatureProvider>)inputs
                                         predictionOptions:(MLPredictionOptions *)predictionOptions
                                               eventLogger:(const executorchcoreml::ModelEventLogger *)eventLogger
                                                     error:(NSError * __autoreleasing *)error {
    if (!self.debugger) {
        self.debugger = [[ETCoreMLModelDebugger alloc] initWithModelAsset:self.modelAsset
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
                eventLogger->log_intermediate_tensors(outputs, self.pathToSymbolNameMap);
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
