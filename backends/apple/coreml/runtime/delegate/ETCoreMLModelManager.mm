//
// ETCoreMLModelManager.mm
//
//  Copyright © 2023 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "ETCoreMLModelManager.h"

#import "ETCoreMLAsset.h"
#import "ETCoreMLAssetManager.h"
#import "ETCoreMLDefaultModelExecutor.h"
#import "ETCoreMLLogging.h"
#import "ETCoreMLModel.h"
#import "ETCoreMLModelCompiler.h"
#import "ETCoreMLModelExecutor.h"
#import "ETCoreMLModelLoader.h"
#import "ETCoreMLStrings.h"
#import "MLModel_Prewarm.h"
#import "MLMultiArray_Copy.h"
#import "inmemory_filesystem_utils.hpp"
#import "model_metadata.h"
#import "multiarray.h"
#import "objc_array_util.h"
#import "serde_json.h"

#import <filesystem>
#import <iostream>
#import <memory>
#import <optional>
#import <os/lock.h>
#import <string>
#import <system_error>
#import <vector>

#if ET_EVENT_TRACER_ENABLED
#import "ETCoreMLModelAnalyzer.h"
#import "ETCoreMLModelDebugInfo.h"
#import "ETCoreMLModelStructurePath.h"
#import "objc_safe_cast.h"
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
    if (@available(macOS 12.3, iOS 15.4, tvOS 15.4, watchOS 8.5, *)) {
        [array1 getBytesWithHandler:^(const void *bytes1, NSInteger __unused size1){
            [array2 getBytesWithHandler:^(const void *bytes2, NSInteger __unused size2) {
                result = (bytes1 == bytes2);
            }];
        }];
    } else {
        result = (array1.dataPointer == array2.dataPointer);
    }
    
    return result;
}

MLPredictionOptions *get_prediction_options(NSArray<MLMultiArray *> *outputs,
                                            NSOrderedSet<NSString *> *output_names,
                                            NSError * __autoreleasing *error) {
    MLPredictionOptions *options = [MLPredictionOptions new];
    if (@available(iOS 16.0, tvOS 16.0, watchOS 9.0, *)) {
        NSMutableDictionary<NSString *, id> *output_backings = [NSMutableDictionary dictionary];
        NSEnumerator<NSString *> *enumerator = [output_names objectEnumerator];
        for (MLMultiArray *output in outputs) {
            NSString *output_name = [enumerator nextObject];
            if (output_name.length == 0) {
                ETCoreMLLogErrorAndSetNSError(error, ETCoreMLErrorCorruptedModel, "Model is broken.");
                return nil;
            }
            output_backings[output_name] = output;
        }
        options.outputBackings = output_backings;
    }
    
    return options;
}

void copy(MLMultiArray *src, MLMultiArray *dst) {
    if (::is_backed_by_same_buffer(src, dst)) {
        return;
    }
    
    [src copyInto:dst];
}

void set_outputs(NSArray<MLMultiArray *> *outputs, NSArray<MLMultiArray *> *model_outputs) {
    NSEnumerator<MLMultiArray *> *enumerator = [model_outputs objectEnumerator];
    for (MLMultiArray *output in outputs) {
        MLMultiArray *model_output = [enumerator nextObject];
        ::copy(model_output, output);
    }
}

std::optional<MultiArray::DataType> get_data_type(MLMultiArrayDataType data_type) {
    switch (data_type) {
        case MLMultiArrayDataTypeFloat16: {
            return MultiArray::DataType::Float16;
        }
        case MLMultiArrayDataTypeFloat32: {
            return MultiArray::DataType::Float32;
        }
        case MLMultiArrayDataTypeFloat64: {
            return MultiArray::DataType::Float64;
        }
        case MLMultiArrayDataTypeInt32: {
            return MultiArray::DataType::Int32;
        }
        default: {
            return std::nullopt;
        }
    }
}

void copy(MLMultiArray *src, executorchcoreml::MultiArray& dst) {
    void (^copy_data)(void *) = ^(void *bytes) {
        if (bytes == dst.data()) {
            return;
        }
            
        MultiArray::MemoryLayout src_layout(
            get_data_type(src.dataType).value(), 
            to_vector<size_t>(src.shape), 
            to_vector<ssize_t>(src.strides)
        );
        MultiArray(const_cast<void *>(bytes), std::move(src_layout)).copy(dst);
    };
    if (@available(macOS 12.3, iOS 15.4, tvOS 15.4, watchOS 8.5, *)) {
        [src getBytesWithHandler:^(const void * _Nonnull bytes, NSInteger size) {
            copy_data(const_cast<void *>(bytes));
        }];
    } else {
        copy_data(src.dataPointer);
    }
}

void set_outputs(std::vector<executorchcoreml::MultiArray>& outputs,
                 NSArray<MLMultiArray *> *model_outputs) {
    NSEnumerator<MLMultiArray *> *enumerator = [model_outputs objectEnumerator];
    for (auto& output : outputs) {
        MLMultiArray *model_output = [enumerator nextObject];
        ::copy(model_output, output);
    }
}

NSData * _Nullable get_file_data(const inmemoryfs::InMemoryFileSystem *inMemoryFS,
                                 NSString *fileName) {
    std::error_code ec;
    const auto& file_path = ::canonical_path(fileName);
    __block auto buffer = inMemoryFS->get_file_content(file_path, ec);
    if (!buffer ||  buffer->size() == 0) {
        return nil;
    }
    
    NSData *file_data = [[NSData alloc] initWithBytesNoCopy:buffer->data()
                                                     length:buffer->size()
                                                deallocator:^(void * _Nonnull __unused bytes, NSUInteger __unused length) {
        buffer.reset();
    }];
    
    return file_data;
}

std::optional<ModelMetadata> get_model_metadata(const inmemoryfs::InMemoryFileSystem *inMemoryFS) {
    NSData *file_data = get_file_data(inMemoryFS, ETCoreMLStrings.metadataFileRelativePath);
    if (!file_data) {
        return std::nullopt;
    }
    
    std::string contents;
    contents.assign(static_cast<const char *>(file_data.bytes), file_data.length);
    ModelMetadata metadata;
    metadata.from_json_string(std::move(contents));
    if (metadata.is_valid()) {
        return metadata;
    }
    
    return std::nullopt;
}

std::optional<ModelMetadata> get_model_metadata_for_method(const inmemoryfs::InMemoryFileSystem *inMemoryFS,
                                                           NSString *methodName) {
    // Load the metadata.json file
    auto metadata_opt = get_model_metadata(inMemoryFS);
    if (!metadata_opt.has_value()) {
        return std::nullopt;
    }
    
    ModelMetadata& metadata = metadata_opt.value();
    
    // If this is a multifunction model and a method name is provided,
    // populate the top-level input_names/output_names from the method's metadata
    if (metadata.is_multifunction() && methodName != nil && methodName.length > 0) {
        std::string method_name_str = [methodName UTF8String];
        const MethodMetadata* method_metadata = metadata.get_method_metadata(method_name_str);
        if (method_metadata != nullptr) {
            metadata.input_names = method_metadata->input_names;
            metadata.output_names = method_metadata->output_names;
        } else {
            // Method not found - fall back to default method if available
            if (!metadata.default_method.empty()) {
                const MethodMetadata* default_metadata = metadata.get_method_metadata(metadata.default_method);
                if (default_metadata != nullptr) {
                    metadata.input_names = default_metadata->input_names;
                    metadata.output_names = default_metadata->output_names;
                }
            }
        }
    }
    
    return metadata;
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
    if (![fm createDirectoryAtURL:dst_url withIntermediateDirectories:YES attributes:@{} error:error]) {
        ETCoreMLLogUnderlyingErrorAndSetNSError(error,
                                                ETCoreMLErrorModelSaveFailed,
                                                local_error,
                                                "Failed to create directory when saving model with identifier = %@.",
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
                                      "Failed to write model files to disk when saving model with identifier = %@.",
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
                                               const executorchcoreml::ModelMetadata& metadata,
                                               NSError * __autoreleasing *error) {
    // Always use the metadata's ordered input/output names for consistency.
    // The pytree flatten order during export determines the correct input order,
    // and metadata captures this order.
    // For multifunction models, all functions share the same input/output names
    // (they differ only in shapes, which are handled by multiArrayConstraint).
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

void add_function_name(std::string& identifier, MLModelConfiguration *configuration) {
    // NOTE: For multifunction CoreML models, we intentionally do NOT include the
    // function name in the cache key. The multifunction model should be compiled
    // only once since it contains ALL functions. The functionName setting on
    // MLModelConfiguration determines which function is invoked at runtime when
    // creating the MLModel from the cached compiled files.
    //
    // Previously this added "_func_{name}" to the identifier, which caused
    // redundant compilations (once per function). Now we compile once and reuse.
    (void)identifier;
    (void)configuration;
}

void add_method_name(std::string& identifier, NSString *methodName) {
    // NOTE: For multifunction CoreML models, we intentionally do NOT include the
    // method name in the cache key. The multifunction model should be compiled
    // only once and shared across all methods/functions. The functionName setting
    // on MLModelConfiguration determines which function is invoked at runtime,
    // but the compiled model is the same for all functions.
    (void)identifier;
    (void)methodName;
}

#if ET_EVENT_TRACER_ENABLED
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

ETCoreMLModelDebugInfo * _Nullable get_model_debug_info(const inmemoryfs::InMemoryFileSystem *inMemoryFS,
                                                        NSError * __autoreleasing *error) {
    NSData *file_data = get_file_data(inMemoryFS, ETCoreMLStrings.debugInfoFileRelativePath);
    if (!file_data) {
        return nil;
    }

    return [ETCoreMLModelDebugInfo modelDebugInfoFromData:file_data error:error];
}

NSString *raw_model_identifier(NSString *identifier) {
    return [NSString stringWithFormat:@"raw_%@", identifier];
}

#endif
} //namespace

@interface ETCoreMLModelManager () {
    os_unfair_lock _lock;
}

@property (nonatomic, readonly, strong) NSFileManager *fileManager;
@property (strong, readonly, nonatomic) ETCoreMLAssetManager* assetManager;
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
    
    __block NSError *localError = nil;
    modelAsset = [self.assetManager assetWithIdentifier:identifier error:&localError];
    if (localError) {
        ETCoreMLLogError(localError,
                         "Failed to retrieve asset with identifier = %@.",
                         identifier);
    }
    
    return modelAsset;
}

- (nullable NSURL *)compiledModelURLWithIdentifier:(NSString *)identifier
                                          modelURL:(nullable NSURL *)modelURL
                                        inMemoryFS:(const inmemoryfs::InMemoryFileSystem*)inMemoryFS
                                            dstURL:(NSURL *)dstURL
                                             error:(NSError * __autoreleasing *)error {
    auto modelAssetType = get_model_asset_type(inMemoryFS);
    if (!modelAssetType) {
        ETCoreMLLogErrorAndSetNSError(error,
                                      ETCoreMLErrorCorruptedModel,
                                      "AOT blob is missing model file.");
        return nil;
    }

    // If modelURL is not provided, write model files to the destination directory (dstURL)
    // and obtain a URL pointing to them. Otherwise, use the provided modelURL.
    modelURL = (modelURL == nil) ? ::write_model_files(dstURL, self.fileManager, identifier, modelAssetType.value(), inMemoryFS, error) : modelURL;
    if (!modelURL) {
        // Failed to generate or locate model files, return nil.
        return nil;
    }

    // Handle based on the type of the model asset.
    switch (modelAssetType.value()) {
        case ModelAssetType::CompiledModel: {
            // Model is already compiled.
            ETCoreMLLogInfo("The model in the pte file is pre-compiled.  Skipping compilation.");
            return modelURL;
        }

        case ModelAssetType::Model: {
            // Compile the model.
            ETCoreMLLogInfo("The model in the pte file is not pre-compiled.  Compiling with a 5 min timeout.");
            NSURL *compiledModelURL = [ETCoreMLModelCompiler compileModelAtURL:modelURL
                                                          maxWaitTimeInSeconds:(5 * 60)
                                                                         error:error];
            // Return the URL of the compiled model or nil if compilation fails.
            return compiledModelURL;
        }
    }
}

- (nullable ETCoreMLAsset *)compiledModelAssetWithMetadata:(const ModelMetadata&)metadata
                                                  modelURL:(nullable NSURL *)modelURL
                                                inMemoryFS:(const inmemoryfs::InMemoryFileSystem*)inMemoryFS
                                                     error:(NSError * __autoreleasing *)error {
    NSString *identifier = @(metadata.identifier.c_str());
    __block ETCoreMLAsset *compiledModelAsset = [self assetWithIdentifier:identifier];
    if (compiledModelAsset) {
        ETCoreMLLogInfo("Cache Hit: Successfully retrieved compiled model with identifier=%@ from the models cache.", identifier);
        return compiledModelAsset;
    }
    
    ETCoreMLLogInfo("Cache Miss: Compiled Model with identifier=%@ was not found in the models cache.", identifier);
    __block NSURL *compiledModelURL;
    [self.assetManager withTemporaryDirectory:^(NSURL * _Nonnull directoryURL) {
        // The directory specified by `directoryURL` is unique and will be automatically cleaned up
        // once the enclosing block completes.
        compiledModelURL = [self compiledModelURLWithIdentifier:identifier
                                                              modelURL:modelURL
                                                            inMemoryFS:inMemoryFS
                                                                dstURL:directoryURL
                                                                 error:error];
        if (compiledModelURL) {
            // Move the compiled model to the asset manager to transfer ownership.
            ETCoreMLLogInfo("Successfully got compiled model with identifier=%@.  Transferring ownership to assetManager.", identifier);
            compiledModelAsset = [self.assetManager storeAssetAtURL:compiledModelURL withIdentifier:identifier error:error];
        }
    }];

    if (!compiledModelAsset) {
        ETCoreMLLogInfo("Failed to transfer ownership of asset with identifier=%@ to assetManager", identifier);
        if (compiledModelURL && [self.fileManager fileExistsAtPath:compiledModelURL.path]) {
            // Log what error was since we now attempt backup path, and previous error is overwritten
            if (error && *error) {
                ETCoreMLLogInfo("error=%@", (*error).localizedDescription);
                *error = nil;
            }
            ETCoreMLLogInfo("Attempting to fall back by loading model without transferring ownership");
            auto backingAsset = Asset::make(compiledModelURL, identifier, self.assetManager.fileManager, error);
            if (backingAsset) {
                compiledModelAsset = [[ETCoreMLAsset alloc] initWithBackingAsset:backingAsset.value()];
            }
        }
    }

    // compiledModelAsset can still be nil if our backup path failed

    return compiledModelAsset;
}

#if ET_EVENT_TRACER_ENABLED
- (nullable ETCoreMLAsset *)modelAssetWithMetadata:(const ModelMetadata&)metadata
                                        inMemoryFS:(const inmemoryfs::InMemoryFileSystem*)inMemoryFS
                                             error:(NSError * __autoreleasing *)error {
    NSString *identifier = @(metadata.identifier.c_str());
    NSString *rawIdentifier = raw_model_identifier(identifier);
    __block ETCoreMLAsset *modelAsset = [self assetWithIdentifier:rawIdentifier];
    if (modelAsset) {
        ETCoreMLLogInfo("Cache Hit: Successfully retrieved model with identifier=%@ from the models cache.", identifier);
    } else {
        ETCoreMLLogInfo("Cache Miss: Model with identifier=%@ was not found in the models cache.", identifier);
    }

    [self.assetManager withTemporaryDirectory:^(NSURL * _Nonnull directoryURL) {
        if (modelAsset) {
            return;
        }

        auto modelAssetType = get_model_asset_type(inMemoryFS);
        if (modelAssetType != ModelAssetType::Model) {
            return;
        }

        // The directory specified by `directoryURL` is unique and will be automatically cleaned up
        // once the enclosing block completes.
        NSURL *modelURL = ::write_model_files(directoryURL,
                                              self.fileManager,
                                              identifier,
                                              modelAssetType.value(),
                                              inMemoryFS,
                                              error);
        if (modelURL) {
            // Move the model to the asset manager to transfer ownership.
            modelAsset = [self.assetManager storeAssetAtURL:modelURL withIdentifier:rawIdentifier error:error];
        }
    }];

    return modelAsset;
}

- (nullable id<ETCoreMLModelExecutor>)modelExecutorWithMetadata:(const ModelMetadata&)metadata
                                                     inMemoryFS:(const inmemoryfs::InMemoryFileSystem*)inMemoryFS
                                                  configuration:(MLModelConfiguration *)configuration
                                                          error:(NSError * __autoreleasing *)error {
    NSError *localError = nil;
    ETCoreMLAsset *modelAsset = [self modelAssetWithMetadata:metadata inMemoryFS:inMemoryFS error:&localError];
    if (localError) {
        if (error) {
            *error = localError;
        }

        return nil;
    }

    ETCoreMLAsset *compiledModelAsset = [self compiledModelAssetWithMetadata:metadata
                                                                    modelURL:modelAsset.contentURL
                                                                  inMemoryFS:inMemoryFS
                                                                       error:error];
    if (!compiledModelAsset) {
        return nil;
    }

    ETCoreMLModelDebugInfo *debug_info = get_model_debug_info(inMemoryFS, error);
    // The analyzer requires both the raw (uncompiled) asset and the compiled model asset to perform analysis.
    return [[ETCoreMLModelAnalyzer alloc] initWithCompiledModelAsset:compiledModelAsset
                                                          modelAsset:modelAsset
                                                      modelDebugInfo:debug_info
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
    ETCoreMLAsset *compiledModelAsset = [self compiledModelAssetWithMetadata:metadata
                                                                    modelURL:nil
                                                                  inMemoryFS:inMemoryFS
                                                                       error:error];
    if (!compiledModelAsset) {
        return nil;
    }

    ETCoreMLModel *model = [ETCoreMLModelLoader loadModelWithCompiledAsset:compiledModelAsset
                                                             configuration:configuration
                                                                  metadata:metadata
                                                                     error:error];
    if (!model) {
        return nil;
    }

    return [[ETCoreMLDefaultModelExecutor alloc] initWithModel:model];
}
#endif


- (nullable id<ETCoreMLModelExecutor>)_modelExecutorWithAOTData:(NSData *)data
                                                   configuration:(MLModelConfiguration *)configuration
                                                      methodName:(nullable NSString *)methodName
                                                           error:(NSError * __autoreleasing *)error {
    using namespace inmemoryfs;
    
    auto buffer = MemoryBuffer::make_unowned(const_cast<void *>(data.bytes), data.length);
    std::unique_ptr<InMemoryFileSystem> inMemoryFS = inmemoryfs::make_from_buffer(std::move(buffer));
    if (!inMemoryFS) {
        ETCoreMLLogErrorAndSetNSError(error,
                                      ETCoreMLErrorCorruptedModel,
                                      "Model data is corrupted.");
        return nil;
    }
    
    // For multifunction models, try to load method-specific metadata first.
    // This ensures we get the correct input/output names for this method.
    std::optional<ModelMetadata> metadata = ::get_model_metadata_for_method(inMemoryFS.get(), methodName);
    if (!metadata) {
        ETCoreMLLogErrorAndSetNSError(error,
                                      ETCoreMLErrorCorruptedMetadata,
                                      "Metadata is invalid or missing.");
        return nil;
    }
    
    auto metadataValue = metadata.value();
    
    // For multifunction CoreML models (ML Programs with multiple functions),
    // we need to set functionName to select the correct function within the model.
    // However, legacy single-function models require functionName to be nil.
    // The metadata's "methods" field indicates if this is a multifunction model.
    if (metadataValue.is_multifunction() && methodName != nil) {
#if defined(__IPHONE_18_0) || defined(__MAC_15_0) || defined(__TVOS_18_0) || defined(__WATCHOS_11_0)
        if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, *)) {
            configuration.functionName = methodName;
        } else {
            ETCoreMLLogErrorAndSetNSError(error,
                                          ETCoreMLErrorCorruptedModel,
                                          "Multifunction CoreML models require iOS 18.0+ / macOS 15.0+.");
            return nil;
        }
#else
        ETCoreMLLogErrorAndSetNSError(error,
                                      ETCoreMLErrorCorruptedModel,
                                      "Multifunction CoreML models require iOS 18.0+ / macOS 15.0+ SDK to build.");
        return nil;
#endif
    }
    
    add_compute_unit(metadataValue.identifier, configuration.computeUnits);
    add_function_name(metadataValue.identifier, configuration);
    add_method_name(metadataValue.identifier, methodName);
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
    return [self loadModelFromAOTData:data
                        configuration:configuration
                           methodName:nil
                                error:error];
}

- (ModelHandle *)loadModelFromAOTData:(NSData*)data
                        configuration:(MLModelConfiguration*)configuration
                           methodName:(nullable NSString*)methodName
                                error:(NSError* __autoreleasing*)error {
    id<ETCoreMLModelExecutor> executor = [self _modelExecutorWithAOTData:data
                                                           configuration:configuration
                                                              methodName:methodName
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

    return [model prewarmAndReturnError:error];
}

- (void)prewarmRecentlyUsedAssetsWithMaxCount:(NSUInteger)maxCount {
    NSError *localError = nil;
    NSArray<ETCoreMLAsset *> *assets = [self.assetManager mostRecentlyUsedAssetsWithMaxCount:maxCount error:&localError];
    
    if (localError) {
        ETCoreMLLogError(localError, "Failed to retrieve recently used assets.");
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
                ETCoreMLLogError(prewarmError,
                                 "Failed to prewarm asset with identifier = %@",
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

- (nullable NSArray<MLMultiArray *> *)executeModelUsingExecutor:(id<ETCoreMLModelExecutor>)executor
                                                         inputs:(NSArray<MLMultiArray *> *)inputs
                                                 outputBackings:(NSArray<MLMultiArray *> *)outputBackings
                                                 loggingOptions:(const executorchcoreml::ModelLoggingOptions&)loggingOptions
                                                    eventLogger:(const executorchcoreml::ModelEventLogger* _Nullable)eventLogger
                                                          error:(NSError * __autoreleasing *)error {
    NSError *localError = nil;
    ETCoreMLModel *model = executor.model;
    MLPredictionOptions *predictionOptions = ::get_prediction_options(outputBackings, model.orderedOutputNames, error);
    if (!predictionOptions) {
        return nil;
    }
    
    id<MLFeatureProvider> inputFeatures = ::get_feature_provider(inputs, model.orderedInputNames, error);
    if (!inputFeatures) {
        return nil;
    }
    
    NSArray<MLMultiArray *> *modelOutputs = [executor executeModelWithInputs:inputFeatures
                                                           predictionOptions:predictionOptions
                                                              loggingOptions:loggingOptions
                                                                 eventLogger:eventLogger
                                                                       error:&localError];
    // Try without output backings.
    if (@available(iOS 16.0, tvOS 16.0, watchOS 9.0, *)) {
        if (!modelOutputs && predictionOptions.outputBackings.count > 0) {
            executor.ignoreOutputBackings = YES;
            localError = nil;
            modelOutputs = [executor executeModelWithInputs:inputFeatures
                                          predictionOptions:predictionOptions
                                             loggingOptions:loggingOptions
                                                eventLogger:eventLogger
                                                      error:&localError];
        }
    }

    if (error) {
        *error = localError;
    }
    
    return modelOutputs;
}

- (BOOL)executeModelWithHandle:(ModelHandle *)handle
                          args:(NSArray<MLMultiArray *> *)args
                loggingOptions:(const executorchcoreml::ModelLoggingOptions&)loggingOptions
                   eventLogger:(const executorchcoreml::ModelEventLogger* _Nullable)eventLogger
                         error:(NSError * __autoreleasing *)error {
    BOOL result = NO;
    id<ETCoreMLModelExecutor> executor = [self executorWithHandle:handle];
    if (!executor) {
        ETCoreMLLogErrorAndSetNSError(error,
                                      ETCoreMLErrorInternalError,
                                      "Model is already unloaded.");
        return result;
    }

    ETCoreMLModel *model = executor.model;
    if (args.count != model.orderedInputNames.count + model.orderedOutputNames.count) {
        ETCoreMLLogErrorAndSetNSError(error,
                                      ETCoreMLErrorCorruptedModel,
                                      "Model is invalid, expected args count to be %lu but got %lu.",
                                      static_cast<unsigned long>(model.orderedInputNames.count + model.orderedOutputNames.count),
                                      args.count);
        return result;
    }

    NSError *localError = nil;
    @autoreleasepool {
        NSArray<MLMultiArray *> *inputs = [args subarrayWithRange:NSMakeRange(0, model.orderedInputNames.count)];
        NSArray<MLMultiArray *> *outputs = [args subarrayWithRange:NSMakeRange(model.orderedInputNames.count, args.count - model.orderedInputNames.count)];
        NSArray<MLMultiArray *> *outputBackings = @[];
        if (executor.ignoreOutputBackings == NO) {
            outputBackings = outputs;
        }
        NSArray<MLMultiArray *> *modelOutputs = [self executeModelUsingExecutor:executor
                                                                         inputs:inputs
                                                                 outputBackings:outputBackings
                                                                 loggingOptions:loggingOptions
                                                                    eventLogger:eventLogger
                                                                          error:&localError];
        if (modelOutputs) {
            ::set_outputs(outputs, modelOutputs);
            result = YES;
        }
    }

    if (localError && error) {
        *error = localError;
    }

    return result;
}

- (BOOL)executeModelWithHandle:(ModelHandle *)handle
                       argsVec:(std::vector<executorchcoreml::MultiArray>&)argsVec
                loggingOptions:(const executorchcoreml::ModelLoggingOptions&)loggingOptions
                   eventLogger:(const executorchcoreml::ModelEventLogger* _Nullable)eventLogger
                         error:(NSError * __autoreleasing *)error {
    BOOL result = NO;
    id<ETCoreMLModelExecutor> executor = [self executorWithHandle:handle];
    if (!executor) {
        ETCoreMLLogErrorAndSetNSError(error,
                                      ETCoreMLErrorInternalError,
                                      "Model is already unloaded.");
        return result;
    }
    ETCoreMLModel *model = executor.model;
    if (argsVec.size() != model.orderedInputNames.count + model.orderedOutputNames.count) {
        ETCoreMLLogErrorAndSetNSError(error,
                                      ETCoreMLErrorCorruptedModel,
                                      "Model is invalid, expected args count to be %lu but got %lu.",
                                      static_cast<unsigned long>(model.orderedInputNames.count + model.orderedOutputNames.count),
                                      argsVec.size());
        return result;
    }
    std::vector<executorchcoreml::MultiArray> inputArgs(argsVec.begin(), argsVec.begin() + model.orderedInputNames.count);
    std::vector<executorchcoreml::MultiArray> outputArgs(argsVec.begin() + model.orderedInputNames.count, argsVec.end());
    NSError *localError = nil;
    @autoreleasepool {
        NSArray<MLMultiArray *> *inputs = [model prepareInputs:inputArgs error:&localError];
        if (inputs) {
            NSArray<MLMultiArray *> *outputBackings = @[];
            if (executor.ignoreOutputBackings == NO) {
                outputBackings = [model prepareOutputBackings:outputArgs error:&localError];
            }
            if (outputBackings) {
                NSArray<MLMultiArray *> *modelOutputs = [self executeModelUsingExecutor:executor
                                                                                 inputs:inputs
                                                                         outputBackings:outputBackings
                                                                         loggingOptions:loggingOptions
                                                                            eventLogger:eventLogger
                                                                                  error:&localError];
                if (modelOutputs) {
                    // Resize for dynamic shapes
                    for (int i = 0; i < outputArgs.size(); i++) {
                        auto new_size = to_vector<size_t>(modelOutputs[i].shape);
                        outputArgs[i].resize(new_size);
                        argsVec[model.orderedInputNames.count + i].resize(new_size);
                    }
                    ::set_outputs(outputArgs, modelOutputs);
                    result = YES;
                }
            }
        }
    }
    if (!result) {
        if (error) {
            *error = localError;
        }
    }
    return result;
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

- (BOOL)purgeModelsCacheAndReturnError:(NSError *__autoreleasing *)error {
    return [self.assetManager purgeAndReturnError:error];
}

@end
