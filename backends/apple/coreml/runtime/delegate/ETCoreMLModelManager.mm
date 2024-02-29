//
// ETCoreMLModelManager.mm
//
//  Copyright © 2023 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "ETCoreMLModelManager.h"

#import <filesystem>
#import <memory>
#import <optional>
#import <string>
#import <system_error>
#import <vector>
#import <iostream>

#import <os/lock.h>

#import <inmemory_filesystem_utils.hpp>
#import <metadata.h>
#import <serde_json.h>

#import <ETCoreMLLogging.h>
#import <ETCoreMLAsset.h>
#import <ETCoreMLAssetManager.h>
#import <ETCoreMLModel.h>
#import <ETCoreMLStrings.h>
#import <MLMultiArray_Copy.h>
#import <MLModel_Prewarm.h>

namespace {

using namespace executorchcoreml;

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
                 id<MLFeatureProvider> feature_provider,
                 NSOrderedSet<NSString *> *output_names,
                 NSError * __autoreleasing *error) {
    NSEnumerator<NSString *> *enumerator = [output_names objectEnumerator];
    for (MLMultiArray *output in outputs) {
        NSString *name = [enumerator nextObject];
        MLFeatureValue *featureValue = [feature_provider featureValueForName:name];
        MLMultiArray *result = featureValue.multiArrayValue;
        if (!::copy(result, output, error)) {
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
    if (metadata.isValid()) {
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

NSURL * _Nullable compile_model(NSURL *model_url, NSError * __autoreleasing *error) {
    __block NSError *local_error = nil;
    __block NSURL *result = nil;
    
    dispatch_semaphore_t sema = dispatch_semaphore_create(0);
    [MLModel compileModelAtURL:model_url completionHandler:^(NSURL * _Nullable temp_url, NSError * _Nullable compilation_error) {
        result = [temp_url copy];
        local_error = compilation_error;
        dispatch_semaphore_signal(sema);
    }];
    
    long status = dispatch_semaphore_wait(sema, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(5 * 60 * NSEC_PER_SEC)));
    if (status != 0) {
        ETCoreMLLogUnderlyingErrorAndSetNSError(error,
                                                ETCoreMLErrorCompilationFailed,
                                                local_error,
                                                "%@: Failed to compile model.",
                                                NSStringFromClass(ETCoreMLModelManager.class));
        return nil;
    }
    return result;
}

NSURL * _Nullable write_model_files(NSURL *dst_url,
                                    NSFileManager *fm,
                                    NSString *identifier,
                                    const inmemoryfs::InMemoryFileSystem& inmemory_fs,
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
    const auto& file_path = canonical_path(ETCoreMLStrings.modelFileRelativePath);
    if (!inmemory_fs.write_item_to_disk(file_path, model_path, true, ec)) {
        ETCoreMLLogErrorAndSetNSError(error,
                                      ETCoreMLErrorModelSaveFailed,
                                      "%@: Failed to write model files to disk when saving model with identifier = %@.",
                                      NSStringFromClass(ETCoreMLModelManager.class),
                                      identifier);
        return nil;
    }
    
    return [dst_url URLByAppendingPathComponent:[NSString stringWithFormat:@"model.%@", ETCoreMLStrings.modelExtensionName]];
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

ETCoreMLModel * _Nullable load_model_from_url(NSURL *compiled_url,
                                              MLModelConfiguration *configuration,
                                              const ModelMetadata& metadata,
                                              ETCoreMLAssetManager *asset_manager,
                                              NSFileManager *fm,
                                              NSError * __autoreleasing *error) {
    NSError *local_error = nil;
    NSString *identifier = @(metadata.identifier.c_str());
    ETCoreMLAsset *asset = [asset_manager storeAssetAtURL:compiled_url withIdentifier:identifier error:&local_error];
    ETCoreMLModel *model = (asset != nil) ? get_model_from_asset(asset, configuration, metadata, &local_error) : nil;
    if (model) {
        return model;
    }
    
    if (local_error) {
        ETCoreMLLogError(local_error,
                         "%@: Failed to load model from asset with identifier = %@",
                         NSStringFromClass(ETCoreMLModelManager.class),
                         identifier);
    }
    
    // If store failed then we will load the model from compiledURL.
    auto backing_asset = Asset::make(compiled_url, identifier, fm, error);
    if (!backing_asset) {
        return nil;
    }
    
    asset = [[ETCoreMLAsset alloc] initWithBackingAsset:backing_asset.value()];
    return get_model_from_asset(asset, configuration, metadata, error);
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
} //namespace

@interface ETCoreMLModelManager () {
    os_unfair_lock _lock;
}

@property (nonatomic, readonly, strong) NSFileManager *fileManager;

@property (nonatomic, readonly, strong) NSMutableDictionary<NSValue *, ETCoreMLModel *> *handleToModelMap;
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
        _handleToModelMap = [NSMutableDictionary dictionary];
        _modelIdentifierToLoadingQueueMap = [NSMapTable strongToWeakObjectsMapTable];
        _modelIdentifierToPrewarmedAssetMap = [NSMutableDictionary dictionary];
        _fileManager = [[NSFileManager alloc] init];
        dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_DEFAULT, -1);
        _prewarmQueue = dispatch_queue_create("com.executorchcoreml.modelmanager.prewarm", attr);
    }
    
    return self;
}

- (nullable ETCoreMLModel *)modelWithHandle:(ModelHandle *)handle {
    ETCoreMLModel *model = nil;
    NSValue *key = [NSValue valueWithPointer:handle];
    {
        os_unfair_lock_lock(&_lock);
        model = self.handleToModelMap[key];
        os_unfair_lock_unlock(&_lock);
    }
    
    return model;
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

- (nullable ETCoreMLModel *)loadModelUsingMetadata:(const ModelMetadata&)metadata
                                        inMemoryFS:(const inmemoryfs::InMemoryFileSystem *)inMemoryFS
                                     configuration:(MLModelConfiguration *)configuration
                                             error:(NSError * __autoreleasing *)error {
    NSString *identifier = @(metadata.identifier.c_str());
    // Otherwise try to retrieve the compiled asset.
    ETCoreMLAsset *asset = [self assetWithIdentifier:identifier];
    ETCoreMLModel *model = asset ? get_model_from_asset(asset, configuration, metadata, error) : nil;
    if (model) {
        return model;
    }
    
    // Create a unique directory for writing model files.
    NSURL *dstURL = [self.assetManager.trashDirectoryURL URLByAppendingPathComponent:[NSUUID UUID].UUIDString];
    // Write the model files for on-device compilation.
    NSURL *modelURL = ::write_model_files(dstURL, self.fileManager, identifier, *inMemoryFS, error);
    if (!modelURL) {
        return nil;
    }
    
    // Compile the model.
    NSURL *compiledModelURL = ::compile_model(modelURL, error);
    if (!compiledModelURL) {
        return nil;
    }
    
    model = ::load_model_from_url(compiledModelURL,
                                  configuration,
                                  metadata,
                                  self.assetManager,
                                  self.fileManager,
                                  error);
    return model;
}

- (nullable ETCoreMLModel *)loadModelFromData:(NSData *)data
                                configuration:(MLModelConfiguration *)configuration
                                        error:(NSError * __autoreleasing *)error {
    using namespace inmemoryfs;
    
    auto buffer = MemoryBuffer::make_unowned(const_cast<void *>(data.bytes), data.length);
    std::unique_ptr<InMemoryFileSystem> inMemoryFS = inmemoryfs::make(buffer);
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
    __block ETCoreMLModel *model = nil;
    dispatch_queue_t loadingQueue = [self queueForLoadingModelWithIdentifier:identifier];
    auto inMemoryFSPtr = inMemoryFS.get();
    dispatch_sync(loadingQueue, ^{
        model = [self loadModelUsingMetadata:metadataValue
                                  inMemoryFS:inMemoryFSPtr
                               configuration:configuration
                                       error:error];
    });
    
    return model;
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

- (ModelHandle *)loadModelFromAOTData:(NSData *)data
                        configuration:(MLModelConfiguration *)configuration
                                error:(NSError * __autoreleasing *)error {
    ETCoreMLModel *model = [self loadModelFromData:data configuration:configuration error:error];
    {
        os_unfair_lock_lock(&_lock);
        if (model) {
            NSValue *key = [NSValue valueWithPointer:(__bridge void *)model];
            self.handleToModelMap[key] = model;
        }
        os_unfair_lock_unlock(&_lock);
    }
    
    return (__bridge ModelHandle *)model;
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
                         error:(NSError * __autoreleasing *)error {
    ETCoreMLModel *model = [self modelWithHandle:handle];
    if (!model) {
        ETCoreMLLogErrorAndSetNSError(error,
                                      0,
                                      "%@: Model is already unloaded.",
                                      NSStringFromClass(self.class));
        return NO;
    }
    
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
    
    id<MLFeatureProvider> outputFeatures = [model.mlModel predictionFromFeatures:inputFeatures
                                                                         options:predictionOptions
                                                                           error:error];
    if (!outputFeatures) {
        return NO;
    }
    
    return ::set_outputs(outputs, outputFeatures, model.orderedOutputNames, error);
}

- (BOOL)unloadModelWithHandle:(ModelHandle *)handle {
    BOOL result = NO;
    @autoreleasepool {
        NSValue *key = [NSValue valueWithPointer:handle];
        os_unfair_lock_lock(&_lock);
        result = (self.handleToModelMap[key] != nil);
        [self.handleToModelMap removeObjectForKey:key];
        os_unfair_lock_unlock(&_lock);
    }
    
    return result;
}

@end
