//
// ETCoreMLModelManager.mm
//
//  Copyright © 2023 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <ETCoreMLAsset.h>
#import <ETCoreMLAssetManager.h>
#import <ETCoreMLDefaultModelExecutor.h>
#import <ETCoreMLLogging.h>
#import <ETCoreMLModel.h>
#import <ETCoreMLModelCompiler.h>
#import <ETCoreMLModelExecutor.h>
#import <ETCoreMLModelLoader.h>
#import <ETCoreMLModelManager.h>
#import <ETCoreMLStrings.h>
#import <MLModel_Prewarm.h>
#import <MLMultiArray_Copy.h>
#import <filesystem>
#import <inmemory_filesystem_utils.hpp>
#import <iostream>
#import <memory>
#import <model_metadata.h>
#import <optional>
#import <os/lock.h>
#import <serde_json.h>
#import <string>
#import <system_error>
#import <vector>

#ifdef ET_EVENT_TRACER_ENABLED
#import <ETCoreMLModelAnalyzer.h>
#endif

namespace {

using namespace executorchcoreml;

enum class ModelAssetType: uint8_t {
    CompiledModel,
    Model
};

std::vector<std::string> canonical_path(NSString *path) {
    NSArray<NSString *> *components = path.pathComponents;
    std::vector<std::string> result;
    result.reserve(components.count);
    for (NSString *component in components) {
        result.emplace_back(component.UTF8String);
    }
    
    return result;
}

id<MLFeatureProvider> _Nullable get_feature_provider(NSArray<MLMultiArray *> *inputs,
                                                     NSOrderedSet<NSString *> *input_names,
                                                     NSError * __autoreleasing *error) {
    NSEnumerator<NSString *> *enumerator = [input_names objectEnumerator];
    NSMutableDictionary<NSString *, MLFeatureValue *> *features = [NSMutableDictionary dictionaryWithCapacity:inputs.count];
    for (MLMultiArray *input in inputs) {
        NSString *input_name = [enumerator nextObject];
        features[input_name] = [MLFeatureValue featureValueWithMultiArray:input];
    }
    
    return [[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:error];
}

BOOL is_backed_by_same_buffer(MLMultiArray *array1, MLMultiArray *array2) {
    __block BOOL result = NO;
    [array1 getBytesWithHandler:^(const void *bytes1, NSInteger __unused size1){
        [array2 getBytesWithHandler:^(const void *bytes2, NSInteger __unused size2) {
            result = (bytes1 == bytes2);
        }];
    }];
    
    return result;
}

MLPredictionOptions *get_prediction_options(NSArray<MLMultiArray *> *outputs,
                                            NSOrderedSet<NSString *> *output_names,
                                            NSError * __autoreleasing *error) {
    MLPredictionOptions *options = [MLPredictionOptions new];
    NSMutableDictionary<NSString *, id> *output_backings = [NSMutableDictionary new];
    NSEnumerator<NSString *> *enumerator = [output_names objectEnumerator];
    for (MLMultiArray *output in outputs) {
        NSString *output_name = [enumerator nextObject];
        if (output_name.length == 0) {
            ETCoreMLLogErrorAndSetNSError(error, 0, "%@: Model is broken.", NSStringFromClass(ETCoreMLModelManager.class));
            return nil;
        }
        output_backings[output_name] = output;
    }
    options.outputBackings = output_backings;
    
    return options;
}

BOOL copy(MLMultiArray *src, MLMultiArray *dst, NSError * __autoreleasing *error) {
    if (![src.shape isEqualToArray:dst.shape]) {
        ETCoreMLLogErrorAndSetNSError(error, 0, "%@: Model is broken", NSStringFromClass(ETCoreMLModelManager.class));
        return NO;
    }
    if (::is_backed_by_same_buffer(src, dst)) {
        return YES;
    }
    @autoreleasepool {
        [src copyInto:dst];
    }
    return YES;
}

BOOL set_outputs(NSArray<MLMultiArray *> *outputs,
                 NSArray<MLMultiArray *> *model_outputs,
                 NSError * __autoreleasing *error) {
    NSEnumerator<MLMultiArray *> *enumerator = [model_outputs objectEnumerator];
    for (MLMultiArray *output in outputs) {
        MLMultiArray *model_output = [enumerator nextObject];
        if (!::copy(output, model_output, error)) {
            return NO;
        }
    }
    
    return YES;
}

std::optional<ModelMetadata> get_model_metadata(const inmemoryfs::InMemoryFileSystem& inMemoryFS) {
    std::error_code error;
    const auto& file_path = ::canonical_path(ETCoreMLStrings.metadataFileRelativePath);
    auto buffer = inMemoryFS.get_file_content(file_path, error);
    if (!buffer) {
        return std::nullopt;
    }
    
    std::string contents;
    contents.assign(reinterpret_cast<char *>(buffer->data()), buffer->size());
    ModelMetadata metadata;
    metadata.from_json_string(std::move(contents));
    if (metadata.is_valid()) {
        return metadata;
    }
    
    return std::nullopt;
}

NSOrderedSet<NSString *> *get_ordered_set(const std::vector<std::string>& values) {
    NSMutableOrderedSet<NSString *> *result = [NSMutableOrderedSet orderedSetWithCapacity:values.size()];
    for (const auto& value : values) {
        [result addObject:@(value.c_str())];
    }
    
    return result;
}

NSURL * _Nullable write_model_files(NSURL *dst_url,
                                    NSFileManager *fm,
                                    NSString *identifier,
                                    ModelAssetType model_asset_type,
                                    const inmemoryfs::InMemoryFileSystem *inmemory_fs,
                                    NSError * __autoreleasing *error) {
    NSError *local_error = nil;
    if (![fm createDirectoryAtURL:dst_url withIntermediateDirectories:NO attributes:@{} error:error]) {
        ETCoreMLLogUnderlyingErrorAndSetNSError(error,
                                                ETCoreMLErrorModelSaveFailed,
                                                local_error,
                                                "%@: Failed to create directory when saving model with identifier = %@.",
                                                NSStringFromClass(ETCoreMLModelManager.class),
                                                identifier);
        return nil;
    }
    
    std::filesystem::path model_path(dst_url.fileSystemRepresentation);
    std::error_code ec;
    std::vector<std::string> file_path;
    switch (model_asset_type) {
        case ModelAssetType::Model: {
            file_path = canonical_path(ETCoreMLStrings.modelFileRelativePath);
            break;
        }
            
        case ModelAssetType::CompiledModel: {
            file_path = canonical_path(ETCoreMLStrings.compiledModelFileRelativePath);
            break;
        }
    }
    
    if (!inmemory_fs->write_item_to_disk(file_path, model_path, true, ec)) {
        ETCoreMLLogErrorAndSetNSError(error,
                                      ETCoreMLErrorModelSaveFailed,
                                      "%@: Failed to write model files to disk when saving model with identifier = %@.",
                                      NSStringFromClass(ETCoreMLModelManager.class),
                                      identifier);
        return nil;
    }
    
    switch (model_asset_type) {
        case ModelAssetType::Model: {
            return [dst_url URLByAppendingPathComponent:[NSString stringWithFormat:@"model.%@", ETCoreMLStrings.modelExtensionName]];
        }
        case ModelAssetType::CompiledModel: {
            return [dst_url URLByAppendingPathComponent:[NSString stringWithFormat:@"model.%@", ETCoreMLStrings.compiledModelExtensionName]];
        }
    }
}

std::optional<ModelAssetType> get_model_asset_type(const inmemoryfs::InMemoryFileSystem *inmemory_fs) {
    std::error_code ec;
    if (inmemory_fs->exists(canonical_path(ETCoreMLStrings.compiledModelFileRelativePath))) {
        return ModelAssetType::CompiledModel;
    }
    
    if (inmemory_fs->exists(canonical_path(ETCoreMLStrings.modelFileRelativePath))) {
        return ModelAssetType::Model;
    }
    
    return std::nullopt;
}


ETCoreMLModel * _Nullable get_model_from_asset(ETCoreMLAsset *asset,
                                               MLModelConfiguration *configuration,
                                               const ModelMetadata& metadata,
                                               NSError * __autoreleasing *error) {
    NSOrderedSet<NSString *> *orderedInputNames = ::get_ordered_set(metadata.input_names);
    NSOrderedSet<NSString *> *orderedOutputNames = ::get_ordered_set(metadata.output_names);
    ETCoreMLModel *model = [[ETCoreMLModel alloc] initWithAsset:asset
                                                  configuration:configuration
                                              orderedInputNames:orderedInputNames
                                             orderedOutputNames:orderedOutputNames
                                                          error:error];
    return model;
}

std::string to_string(MLComputeUnits compute_units) {
    switch (compute_units) {
        case MLComputeUnitsAll: {
            return ETCoreMLStrings.allComputeUnitsName.UTF8String;
        }
        case MLComputeUnitsCPUOnly: {
            return ETCoreMLStrings.cpuComputeUnitName.UTF8String;
        }
        case MLComputeUnitsCPUAndGPU: {
            return ETCoreMLStrings.cpuAndGpuComputeUnitsName.UTF8String;
        }
        case MLComputeUnitsCPUAndNeuralEngine: {
            return ETCoreMLStrings.cpuAndNeuralEngineComputeUnitsName.UTF8String;
        }
        default: {
            return ETCoreMLStrings.allComputeUnitsName.UTF8String;
        }
    }
}

void add_compute_unit(std::string& identifier, MLComputeUnits compute_units) {
    identifier.append("_");
    identifier.append(to_string(compute_units));
}

ETCoreMLAsset * _Nullable make_asset(NSURL *url,
                                     NSString *identifier,
                                     NSFileManager *fm,
                                     NSError * __autoreleasing *error) {
    auto backingAsset = executorchcoreml::Asset::make(url, identifier, fm, error);
    if (!backingAsset) {
        return nil;
    }
    
    return [[ETCoreMLAsset alloc] initWithBackingAsset:std::move(backingAsset.value())];
}
} //namespace

@interface ETCoreMLModelManager () {
    os_unfair_lock _lock;
}

@property (nonatomic, readonly, strong) NSFileManager *fileManager;
@property (nonatomic, readonly, strong) NSMutableDictionary<NSValue *, id<ETCoreMLModelExecutor>> *handleToExecutorMap;
@property (nonatomic, readonly, strong) NSMapTable<NSString *, dispatch_queue_t> *modelIdentifierToLoadingQueueMap;
@property (nonatomic, readonly, strong) NSMutableDictionary<NSString *, ETCoreMLAsset *> *modelIdentifierToPrewarmedAssetMap;
@property (nonatomic, readonly, strong) dispatch_queue_t prewarmQueue;

@end

@implementation ETCoreMLModelManager

- (instancetype)initWithAssetManager:(ETCoreMLAssetManager *)assetManager {
    self = [super init];
    if (self) {
        _assetManager = assetManager;
        _lock = OS_UNFAIR_LOCK_INIT;
        _handleToExecutorMap = [NSMutableDictionary dictionary];
        _modelIdentifierToLoadingQueueMap = [NSMapTable strongToWeakObjectsMapTable];
        _modelIdentifierToPrewarmedAssetMap = [NSMutableDictionary dictionary];
        _fileManager = [[NSFileManager alloc] init];
        dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_DEFAULT, -1);
        _prewarmQueue = dispatch_queue_create("com.executorchcoreml.modelmanager.prewarm", attr);
    }
    
    return self;
}

- (nullable id<ETCoreMLModelExecutor>)executorWithHandle:(ModelHandle *)handle {
    id<ETCoreMLModelExecutor> executor = nil;
    NSValue *key = [NSValue valueWithPointer:handle];
    {
        os_unfair_lock_lock(&_lock);
        executor = self.handleToExecutorMap[key];
        os_unfair_lock_unlock(&_lock);
    }
    
    return executor;
}

- (nullable ETCoreMLModel *)modelWithHandle:(ModelHandle *)handle {
    id<ETCoreMLModelExecutor> executor = [self executorWithHandle:handle];
    return executor.model;
}

- (nullable ETCoreMLAsset *)assetWithIdentifier:(NSString *)identifier {
    ETCoreMLAsset *modelAsset = nil;
    {
        os_unfair_lock_lock(&_lock);
        modelAsset = self.modelIdentifierToPrewarmedAssetMap[identifier];
        os_unfair_lock_unlock(&_lock);
    }
    
    if (modelAsset) {
        return modelAsset;
    }
    
    NSError *localError = nil;
    modelAsset = [self.assetManager assetWithIdentifier:identifier error:&localError];
    if (localError) {
        ETCoreMLLogError(localError,
                         "%@: Failed to retrieve asset with identifier = %@",
                         NSStringFromClass(self.assetManager.class),
                         identifier);
    }
    
    return modelAsset;
}

- (nullable NSURL *)compiledModelURLWithIdentifier:(NSString *)identifier
                                        inMemoryFS:(const inmemoryfs::InMemoryFileSystem*)inMemoryFS
                                      assetManager:(ETCoreMLAssetManager *)assetManager
                                             error:(NSError * __autoreleasing *)error {
    auto modelAssetType = get_model_asset_type(inMemoryFS);
    if (!modelAssetType) {
        ETCoreMLLogErrorAndSetNSError(error,
                                      ETCoreMLErrorCorruptedModel,
                                      "%@: AOT blob is missing model file.",
                                      NSStringFromClass(ETCoreMLModelManager.class));
        return nil;
    }
    
    NSURL *dstURL = [self.assetManager.trashDirectoryURL URLByAppendingPathComponent:[NSUUID UUID].UUIDString];
    NSURL *modelURL = ::write_model_files(dstURL, self.fileManager, identifier, modelAssetType.value(), inMemoryFS, error);
    switch (modelAssetType.value()) {
        case ModelAssetType::CompiledModel: {
            return modelURL;
        }
            
        case ModelAssetType::Model: {
            // we need to compiled the model.
            NSURL *compiledModelURL = [ETCoreMLModelCompiler compileModelAtURL:modelURL
                                                          maxWaitTimeInSeconds:(5 * 60)
                                                                         error:error];
            
            return compiledModelURL;
        }
    }
}

#if ET_EVENT_TRACER_ENABLED
- (nullable id<ETCoreMLModelExecutor>)modelExecutorWithMetadata:(const ModelMetadata&)metadata
                                                     inMemoryFS:(const inmemoryfs::InMemoryFileSystem*)inMemoryFS
                                                  configuration:(MLModelConfiguration *)configuration
                                                          error:(NSError * __autoreleasing *)error {
    NSString *identifier = @(metadata.identifier.c_str());
    // Otherwise try to retrieve the compiled asset.
    ETCoreMLAsset *compiledModelAsset = [self assetWithIdentifier:identifier];
    // Create a unique directory for writing model files.
    NSURL *dstURL = [self.assetManager.trashDirectoryURL URLByAppendingPathComponent:[NSUUID UUID].UUIDString];
    auto modelAssetType = get_model_asset_type(inMemoryFS);
    ETCoreMLAsset *modelAsset = nil;
    // Write the model files.
    if (modelAssetType == ModelAssetType::ModelPackage) {
        NSURL *modelURL = ::write_model_files(dstURL, self.fileManager, identifier, modelAssetType.value(), inMemoryFS, error);
        if (modelURL) {
            modelAsset = make_asset(modelURL,
                                    identifier,
                                    self.fileManager,
                                    error);
        }
    }
   
    if (!compiledModelAsset) {
        // Compile the model.
        NSURL *compiledModelURL = [self compiledModelURLWithIdentifier:identifier
                                                            inMemoryFS:inMemoryFS
                                                          assetManager:self.assetManager
                                                                 error:error];
        compiledModelAsset = make_asset(compiledModelURL,
                                        identifier,
                                        self.fileManager,
                                        error);
    }
    
    if (!compiledModelAsset) {
        return nil;
    }
    

    return [[ETCoreMLModelAnalyzer alloc] initWithCompiledModelAsset:compiledModelAsset
                                                          modelAsset:modelAsset
                                                            metadata:metadata
                                                       configuration:configuration
                                                        assetManager:self.assetManager
                                                               error:error];
}

#else
- (nullable id<ETCoreMLModelExecutor>)modelExecutorWithMetadata:(const ModelMetadata&)metadata
                                                     inMemoryFS:(const inmemoryfs::InMemoryFileSystem*)inMemoryFS
                                                  configuration:(MLModelConfiguration *)configuration
                                                          error:(NSError * __autoreleasing *)error {
    NSString *identifier = @(metadata.identifier.c_str());
    // Otherwise try to retrieve the compiled asset.
    ETCoreMLAsset *asset = [self assetWithIdentifier:identifier];
    ETCoreMLModel *model = asset ? get_model_from_asset(asset, configuration, metadata, error) : nil;
    if (model) {
        return [[ETCoreMLDefaultModelExecutor alloc] initWithModel:model];
    }
    
    // Compile the model.
    NSURL *compiledModelURL = [self compiledModelURLWithIdentifier:identifier
                                                        inMemoryFS:inMemoryFS
                                                      assetManager:self.assetManager
                                                             error:error];
    if (!compiledModelURL) {
        return nil;
    }
    
    model = [ETCoreMLModelLoader loadModelWithContentsOfURL:compiledModelURL
                                              configuration:configuration
                                                   metadata:metadata
                                               assetManager:self.assetManager
                                                      error:error];
    
    return [[ETCoreMLDefaultModelExecutor alloc] initWithModel:model];
}
#endif

- (nullable id<ETCoreMLModelExecutor>)_modelExecutorWithAOTData:(NSData *)data
                                                  configuration:(MLModelConfiguration *)configuration
                                                          error:(NSError * __autoreleasing *)error {
    using namespace inmemoryfs;
    
    auto buffer = MemoryBuffer::make_unowned(const_cast<void *>(data.bytes), data.length);
    std::unique_ptr<InMemoryFileSystem> inMemoryFS = inmemoryfs::make_from_buffer(std::move(buffer));
    if (!inMemoryFS) {
        ETCoreMLLogErrorAndSetNSError(error,
                                      ETCoreMLErrorCorruptedModel,
                                      "%@: Model data is corrupted.",
                                      NSStringFromClass(ETCoreMLModelManager.class));
        return nil;
    }
    
    std::optional<ModelMetadata> metadata = ::get_model_metadata(*inMemoryFS);
    if (!metadata) {
        ETCoreMLLogErrorAndSetNSError(error,
                                      ETCoreMLErrorCorruptedMetadata,
                                      "%@: Metadata is invalid or missing.",
                                      NSStringFromClass(ETCoreMLModelManager.class));
        return nil;
    }
    
    auto metadataValue = metadata.value();
    add_compute_unit(metadataValue.identifier, configuration.computeUnits);
    NSString *identifier = @(metadataValue.identifier.c_str());
    // If there are multiple calls to load the same model, we only want to compile it once.
    __block id<ETCoreMLModelExecutor> executor = nil;
    dispatch_queue_t loadingQueue = [self queueForLoadingModelWithIdentifier:identifier];
    auto inMemoryFSPtr = inMemoryFS.get();
    dispatch_sync(loadingQueue, ^{
        executor = [self modelExecutorWithMetadata:metadataValue
                                        inMemoryFS:inMemoryFSPtr
                                     configuration:configuration
                                             error:error];
    });
    
    return executor;
}

- (dispatch_queue_t)queueForLoadingModelWithIdentifier:(NSString *)identifier {
    os_unfair_lock_lock(&_lock);
    dispatch_queue_t queue = [self.modelIdentifierToLoadingQueueMap objectForKey:identifier];
    if (!queue) {
        queue = dispatch_queue_create("com.executorchcoreml.modelmanager.loader", DISPATCH_QUEUE_SERIAL_WITH_AUTORELEASE_POOL);
        [self.modelIdentifierToLoadingQueueMap setObject:queue forKey:identifier];
    }
    os_unfair_lock_unlock(&_lock);
    
    return queue;
}

- (ModelHandle *)loadModelFromAOTData:(NSData*)data
                        configuration:(MLModelConfiguration*)configuration
                                error:(NSError* __autoreleasing*)error {
    id<ETCoreMLModelExecutor> executor = [self _modelExecutorWithAOTData:data
                                                           configuration:configuration
                                                                   error:error];
    {
        os_unfair_lock_lock(&_lock);
        if (executor) {
            NSValue *key = [NSValue valueWithPointer:(__bridge void *)executor.model];
            self.handleToExecutorMap[key] = executor;
        }
        os_unfair_lock_unlock(&_lock);
    }
    
    return (__bridge ModelHandle *)executor.model;
}

- (BOOL)prewarmModelWithHandle:(ModelHandle *)handle
                         error:(NSError * __autoreleasing *)error {
    ETCoreMLModel *model = [self modelWithHandle:handle];
    if (!model) {
        return NO;
    }
    
    NSError *localError = nil;
    BOOL result = [model.mlModel prewarmAndReturnError:&localError];
    if (!result) {
        ETCoreMLLogError(localError,
                         "%@: Failed to prewarm model with identifier = %@",
                         NSStringFromClass(self.assetManager.class),
                         model.identifier);
    }
    
    if (error) {
        *error = localError;
    }
    
    return result;
}

- (void)prewarmRecentlyUsedAssetsWithMaxCount:(NSUInteger)maxCount {
    NSError *localError = nil;
    NSArray<ETCoreMLAsset *> *assets = [self.assetManager mostRecentlyUsedAssetsWithMaxCount:maxCount error:&localError];
    
    if (localError) {
        ETCoreMLLogError(localError,
                         "%@: Failed to retrieve recently used assets.",
                         NSStringFromClass(self.assetManager.class));
    }
    
    if (assets.count == 0) {
        return;
    }
    
    for (ETCoreMLAsset *asset in assets) {
        __weak __typeof(self) weakSelf = self;
        dispatch_async(self.prewarmQueue, ^{
            __strong __typeof(self) strongSelf = weakSelf;
            if (!strongSelf) {
                return;
            }
            
            NSError *prewarmError = nil;
            if (![asset prewarmAndReturnError:&prewarmError]) {
                ETCoreMLLogError(localError,
                                 "%@: Failed to prewarm asset with identifier = %@",
                                 NSStringFromClass(strongSelf.assetManager.class),
                                 asset.identifier);
                return;
            }
            
            [strongSelf addPrewarmedAsset:asset];
        });
    }
}

- (void)addPrewarmedAsset:(ETCoreMLAsset *)asset {
    os_unfair_lock_lock(&_lock);
    [self.modelIdentifierToPrewarmedAssetMap setObject:asset forKey:asset.identifier];
    os_unfair_lock_unlock(&_lock);
}

- (BOOL)executeModelWithHandle:(ModelHandle *)handle
                          args:(NSArray<MLMultiArray *> *)args
                loggingOptions:(const executorchcoreml::ModelLoggingOptions&)loggingOptions
                   eventLogger:(const executorchcoreml::ModelEventLogger* _Nullable)eventLogger
                         error:(NSError * __autoreleasing *)error {
    id<ETCoreMLModelExecutor> executor = [self executorWithHandle:handle];
    if (!executor) {
        ETCoreMLLogErrorAndSetNSError(error,
                                      0,
                                      "%@: Model is already unloaded.",
                                      NSStringFromClass(self.class));
        return NO;
    }
    
    ETCoreMLModel *model = executor.model;
    if (args.count != model.orderedInputNames.count + model.orderedOutputNames.count) {
        ETCoreMLLogErrorAndSetNSError(error,
                                      ETCoreMLErrorCorruptedModel,
                                      "%@: Model is invalid.",
                                      NSStringFromClass(self.class));
        return NO;
    }
    
    NSArray<MLMultiArray *> *inputs = [args subarrayWithRange:NSMakeRange(0, model.orderedInputNames.count)];
    NSArray<MLMultiArray *> *outputs = [args subarrayWithRange:NSMakeRange(model.orderedInputNames.count, args.count - model.orderedInputNames.count)];
    id<MLFeatureProvider> inputFeatures = ::get_feature_provider(inputs, model.orderedInputNames, error);
    if (!inputFeatures) {
        return NO;
    }
    
    MLPredictionOptions *predictionOptions = ::get_prediction_options(outputs, model.orderedOutputNames, error);
    if (!predictionOptions) {
        return NO;
    }
    
    NSArray<MLMultiArray *> *modelOutputs = [executor executeModelWithInputs:inputFeatures
                                                           predictionOptions:predictionOptions
                                                             loggingOptions:loggingOptions
                                                                 eventLogger:eventLogger
                                                                       error:error];
    if (!outputs) {
        return NO;
    }
    
    return ::set_outputs(outputs, modelOutputs, error);
}

- (BOOL)unloadModelWithHandle:(ModelHandle *)handle {
    BOOL result = NO;
    @autoreleasepool {
        NSValue *key = [NSValue valueWithPointer:handle];
        os_unfair_lock_lock(&_lock);
        result = (self.handleToExecutorMap[key] != nil);
        [self.handleToExecutorMap removeObjectForKey:key];
        os_unfair_lock_unlock(&_lock);
    }
    
    return result;
}

@end
