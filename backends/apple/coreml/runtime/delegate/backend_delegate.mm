//
// backend_delegate.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.


#import "backend_delegate.h"

#import "ETCoreMLAssetManager.h"
#import "ETCoreMLLogging.h"
#import "ETCoreMLModel.h"
#import "ETCoreMLModelCache.h"
#import "ETCoreMLModelManager.h"
#import "ETCoreMLStrings.h"
#import "model_event_logger.h"
#import "multiarray.h"

namespace  {
using namespace executorchcoreml;

std::optional<MLComputeUnits> get_compute_units(const Buffer& buffer) {
    std::string value(reinterpret_cast<const char *>(buffer.data()), buffer.size());
    if (value == std::string(ETCoreMLStrings.cpuComputeUnitName.UTF8String)) {
        return MLComputeUnitsCPUOnly;
    } else if (value == std::string(ETCoreMLStrings.cpuAndGpuComputeUnitsName.UTF8String)) {
        return MLComputeUnitsCPUAndGPU;
    } else if (value == std::string(ETCoreMLStrings.cpuAndNeuralEngineComputeUnitsName.UTF8String)) {
        return MLComputeUnitsCPUAndNeuralEngine;
    } else if (value == std::string(ETCoreMLStrings.allComputeUnitsName.UTF8String)) {
        return MLComputeUnitsAll;
    } else {
        return std::nullopt;
    }
}

MLModelConfiguration * _Nullable get_model_configuration(const std::unordered_map<std::string, Buffer>& specs,
                                                         NSError * __autoreleasing *error) {
    std::string compute_units_key(ETCoreMLStrings.computeUnitsKeyName.UTF8String);
    MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];

    for (const auto& [key, buffer] : specs) {
        if (key == compute_units_key) {
            auto compute_units = get_compute_units(buffer);
            if (!compute_units.has_value()) {
                std::string value(reinterpret_cast<const char *>(buffer.data()), buffer.size());
                NSString *errorMessage = [NSString stringWithFormat:@"Invalid compute_unit value: '%s'. Valid values are: %@, %@, %@, %@",
                    value.c_str(),
                    ETCoreMLStrings.cpuComputeUnitName,
                    ETCoreMLStrings.cpuAndGpuComputeUnitsName,
                    ETCoreMLStrings.cpuAndNeuralEngineComputeUnitsName,
                    ETCoreMLStrings.allComputeUnitsName];
                if (error) {
                    *error = [NSError errorWithDomain:ETCoreMLStrings.productIdentifier
                                                 code:-1
                                             userInfo:@{NSLocalizedDescriptionKey: errorMessage}];
                }
                return nil;
            }
            configuration.computeUnits = compute_units.value();
            break;
        }
    }

    return configuration;
}

ETCoreMLAssetManager * _Nullable create_asset_manager(NSString *assets_directory_path,
                                                      NSString *trash_directory_path,
                                                      NSString *database_directory_path,
                                                      NSString *database_name,
                                                      NSInteger max_assets_size_in_bytes,
                                                      NSError * __autoreleasing *error) {
    NSURL *assets_directory_url = [NSURL fileURLWithPath:assets_directory_path];
    NSURL *trash_directory_url = [NSURL fileURLWithPath:trash_directory_path];
    NSURL *database_directory_url = [NSURL fileURLWithPath:database_directory_path];
    NSURL *database_url = [database_directory_url URLByAppendingPathComponent:database_name];
    ETCoreMLAssetManager *manager = [[ETCoreMLAssetManager alloc] initWithDatabaseURL:database_url
                                                                   assetsDirectoryURL:assets_directory_url
                                                                    trashDirectoryURL:trash_directory_url
                                                                 maxAssetsSizeInBytes:max_assets_size_in_bytes
                                                                                error:error];
    return manager;
}
} //namespace

@interface ETCoreMLModelManagerDelegate : NSObject

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

- (instancetype)initWithConfig:(BackendDelegate::Config)config NS_DESIGNATED_INITIALIZER;

- (BOOL)loadAndReturnError:(NSError * _Nullable __autoreleasing *)error;

- (void)loadAsynchronously;

- (ModelHandle*)loadModelFromAOTData:(NSData*)data
                       configuration:(MLModelConfiguration*)configuration
                          methodName:(nullable NSString*)methodName
                        functionName:(nullable NSString*)functionName
                                error:(NSError* __autoreleasing*)error;

- (ModelHandle*)loadModelFromAOTData:(NSData*)data
                       configuration:(MLModelConfiguration*)configuration
                          methodName:(nullable NSString*)methodName
                        functionName:(nullable NSString*)functionName
                           cachePath:(nullable NSString*)cachePath
                                error:(NSError* __autoreleasing*)error;

- (ModelHandle*)loadModelFromAOTData:(NSData*)data
                       configuration:(MLModelConfiguration*)configuration
                               error:(NSError* __autoreleasing*)error;

- (BOOL)executeModelWithHandle:(ModelHandle*)handle
                       argsVec:(std::vector<executorchcoreml::MultiArray>&)argsVec
                loggingOptions:(const executorchcoreml::ModelLoggingOptions&)loggingOptions
                   eventLogger:(const executorchcoreml::ModelEventLogger* _Nullable)eventLogger
                         error:(NSError* __autoreleasing*)error;

- (BOOL)unloadModelWithHandle:(ModelHandle*)handle;

- (BOOL)purgeModelsCacheAndReturnError:(NSError * _Nullable __autoreleasing *)error;

@property (assign, readonly, nonatomic) BackendDelegate::Config config;
@property (strong, readonly, nonatomic) dispatch_queue_t syncQueue;
@property (strong, nonatomic, nullable) ETCoreMLModelManager *impl;
@property (strong, nonatomic, nullable) ETCoreMLModelCache *defaultCache;
@property (assign, readonly, nonatomic) BOOL isAvailable;

@end

@implementation ETCoreMLModelManagerDelegate

- (instancetype)initWithConfig:(BackendDelegate::Config)config {
    self = [super init];
    if (self) {
        _config = std::move(config);
        _syncQueue = dispatch_queue_create("com.executorchcoreml.modelmanagerdelegate.sync", DISPATCH_QUEUE_SERIAL_WITH_AUTORELEASE_POOL);
    }

    return self;
}

- (BOOL)_loadAndReturnError:(NSError * _Nullable __autoreleasing *)error {
    if (self.impl != nil) {
        return YES;
    }

    ETCoreMLAssetManager *assetManager = create_asset_manager(ETCoreMLStrings.assetsDirectoryPath,
                                                              ETCoreMLStrings.trashDirectoryPath,
                                                              ETCoreMLStrings.databaseDirectoryPath,
                                                              ETCoreMLStrings.databaseName,
                                                              self.config.max_models_cache_size,
                                                              error);
    if (!assetManager) {
        return NO;
    }

    ETCoreMLModelManager *modelManager = [[ETCoreMLModelManager alloc] initWithAssetManager:assetManager];
    if (!modelManager) {
        return NO;
    }

    self.impl = modelManager;

    // Create default filesystem cache at the same location as assets
    NSURL *defaultCacheURL = [NSURL fileURLWithPath:ETCoreMLStrings.assetsDirectoryPath isDirectory:YES];
    ETCoreMLModelCache *defaultCache = [[ETCoreMLModelCache alloc] initWithCacheRootDirectory:defaultCacheURL];
    if (defaultCache.isReady) {
        self.defaultCache = defaultCache;
    } else {
        ETCoreMLLogError(defaultCache.initializationError,
                         "Default cache initialization failed, will use asset manager as fallback");
    }

    if (self.config.should_prewarm_asset) {
        [modelManager prewarmRecentlyUsedAssetsWithMaxCount:1];
    }

    return YES;
}

- (BOOL)loadAndReturnError:(NSError * _Nullable __autoreleasing *)error {
    __block NSError *localError = nil;
    __block BOOL result = NO;
    dispatch_sync(self.syncQueue, ^{
        result = [self _loadAndReturnError:&localError];
    });

    if (error) {
        *error = localError;
    }

    return result;
}

- (void)loadAsynchronously {
    dispatch_async(self.syncQueue, ^{
        (void)[self _loadAndReturnError:nil];
    });
}

- (ModelHandle*)loadModelFromAOTData:(NSData*)data
                       configuration:(MLModelConfiguration*)configuration
                                error:(NSError* __autoreleasing*)error {
    return [self loadModelFromAOTData:data
                        configuration:configuration
                           methodName:nil
                         functionName:nil
                            cachePath:nil
                                error:error];
}

- (ModelHandle*)loadModelFromAOTData:(NSData*)data
                       configuration:(MLModelConfiguration*)configuration
                          methodName:(nullable NSString*)methodName
                        functionName:(nullable NSString*)functionName
                               error:(NSError* __autoreleasing*)error {
    return [self loadModelFromAOTData:data
                        configuration:configuration
                           methodName:methodName
                         functionName:functionName
                            cachePath:nil
                                error:error];
}

- (ModelHandle*)loadModelFromAOTData:(NSData*)data
                       configuration:(MLModelConfiguration*)configuration
                          methodName:(nullable NSString*)methodName
                        functionName:(nullable NSString*)functionName
                           cachePath:(nullable NSString*)cachePath
                               error:(NSError* __autoreleasing*)error {
    // Default to using the old cache (useNewCache = NO)
    return [self loadModelFromAOTData:data
                        configuration:configuration
                           methodName:methodName
                         functionName:functionName
                            cachePath:cachePath
                          useNewCache:NO
                                error:error];
}

- (ModelHandle*)loadModelFromAOTData:(NSData*)data
                       configuration:(MLModelConfiguration*)configuration
                          methodName:(nullable NSString*)methodName
                        functionName:(nullable NSString*)functionName
                           cachePath:(nullable NSString*)cachePath
                         useNewCache:(BOOL)useNewCache
                               error:(NSError* __autoreleasing*)error {
    if (![self loadAndReturnError:error]) {
        return nil;
    }

    id<ETCoreMLCache> cache = nil;
    if (cachePath != nil) {
        // Use NEW filesystem cache at specified path
        NSURL *cacheURL = [NSURL fileURLWithPath:cachePath isDirectory:YES];
        ETCoreMLModelCache *modelCache = [[ETCoreMLModelCache alloc] initWithCacheRootDirectory:cacheURL];
        if (!modelCache.isReady) {
            // Fallback error if initializationError is unexpectedly nil
            NSError *cacheError = modelCache.initializationError
                ?: [NSError errorWithDomain:ETCoreMLModelCacheErrorDomain
                                       code:ETCoreMLModelCacheErrorCodeInitializationFailed
                                   userInfo:@{NSLocalizedDescriptionKey: @"Cache initialization failed"}];
            if (error) *error = cacheError;
            return nil;
        }
        cache = modelCache;
    } else if (useNewCache) {
        if (self.defaultCache != nil) {
            // Use default filesystem cache
            cache = self.defaultCache;
} else {
            // Fallback: useNewCache requested but default cache unavailable
            NSError *fallbackError = [NSError errorWithDomain:ETCoreMLErrorDomain
                                                         code:ETCoreMLErrorInternalError
                                                     userInfo:@{NSLocalizedDescriptionKey: @"Default cache unavailable"}];
            ETCoreMLLogError(fallbackError,
                             "useNewCache=YES but default cache is unavailable, falling back to asset manager");
        }
    }
    // If useNewCache is false or defaultCache is nil, cache remains nil
    // and loadModelFromAOTData will use the asset manager path

    auto handle = [self.impl loadModelFromAOTData:data
                                    configuration:configuration
                                       methodName:methodName
                                     functionName:functionName
                                            cache:cache
                                            error:error];
    if ((handle != NULL) && self.config.should_prewarm_model) {
        [self.impl prewarmModelWithHandle:handle error:nil];
    }

    return handle;
}

- (BOOL)executeModelWithHandle:(ModelHandle*)handle
                       argsVec:(std::vector<executorchcoreml::MultiArray>&)argsVec
                loggingOptions:(const executorchcoreml::ModelLoggingOptions&)loggingOptions
                   eventLogger:(const executorchcoreml::ModelEventLogger* _Nullable)eventLogger
                         error:(NSError* __autoreleasing*)error {
    assert(self.impl != nil && "Impl must not be nil");
    return [self.impl executeModelWithHandle:handle
                                     argsVec:argsVec
                              loggingOptions:loggingOptions
                                 eventLogger:eventLogger
                                       error:error];
}

- (nullable ETCoreMLModel*)modelWithHandle:(ModelHandle*)handle {
    assert(self.impl != nil && "Impl must not be nil");
    return [self.impl modelWithHandle:handle];
}

- (BOOL)unloadModelWithHandle:(ModelHandle*)handle {
    assert(self.impl != nil && "Impl must not be nil");
    return [self.impl unloadModelWithHandle:handle];
}

- (BOOL)purgeModelsCacheAndReturnError:(NSError * _Nullable __autoreleasing *)error {
    if (![self loadAndReturnError:error]) {
        return NO;
    }

    return [self.impl purgeModelsCacheAndReturnError:error];;
}

- (BOOL)isAvailable {
    if (![self loadAndReturnError:nil]) {
        return NO;
    }

    return YES;
}

@end

namespace executorchcoreml {

std::string BackendDelegate::ErrorCategory::message(int code) const {
    switch (static_cast<ErrorCode>(code)) {
        case ErrorCode::CorruptedData:
            return "AOT blob can't be parsed";
        case ErrorCode::CorruptedMetadata:
            return "AOT blob has incorrect or missing metadata.";
        case ErrorCode::CorruptedModel:
            return "AOT blob has incorrect or missing CoreML model.";
        case ErrorCode::BrokenModel:
            return "CoreML model doesn't match the input and output specifications.";
        case ErrorCode::CompilationFailed:
            return "Failed to compile CoreML model.";
        case ErrorCode::ModelSaveFailed:
            return "Failed to write CoreML model to disk.";
        case ErrorCode::ModelCacheCreationFailed:
            return "Failed to create model cache.";
        default:
            return "Unexpected error.";
    }
}

class BackendDelegateImpl: public BackendDelegate {
public:
    explicit BackendDelegateImpl(const Config& config) noexcept
    :BackendDelegate(), model_manager_([[ETCoreMLModelManagerDelegate alloc] initWithConfig:config])
    {
        [model_manager_ loadAsynchronously];
    }

    BackendDelegateImpl(BackendDelegateImpl const&) = delete;
    BackendDelegateImpl& operator=(BackendDelegateImpl const&) = delete;

Handle *init(Buffer processed,
             const std::unordered_map<std::string, Buffer>& specs,
             const char* method_name = nullptr,
             const char* function_name = nullptr,
             executorch::runtime::Span<const executorch::runtime::BackendOption> runtime_specs = {}) const noexcept override {
        NSError *localError = nil;
        MLModelConfiguration *configuration = get_model_configuration(specs, &localError);
        if (configuration == nil) {
            ETCoreMLLogError(localError, "Invalid model configuration");
            return nullptr;
        }

        NSString *methodNameStr = method_name ? @(method_name) : nil;
        NSString *functionNameStr = function_name ? @(function_name) : nil;

        // Parse cache_dir and _use_new_cache from runtime_specs
        NSString *cachePath = nil;
        BOOL useNewCache = NO; // Default to using the old cache (asset manager)
        for (size_t i = 0; i < runtime_specs.size(); ++i) {
            const auto& opt = runtime_specs[i];
            if (std::strcmp(opt.key, "cache_dir") == 0) {
                if (auto* arr = std::get_if<std::array<char, executorch::runtime::kMaxOptionValueLength>>(&opt.value)) {
                    cachePath = @(arr->data());
                }
            } else if (std::strcmp(opt.key, "_use_new_cache") == 0) {
                if (auto* val = std::get_if<bool>(&opt.value)) {
                    useNewCache = *val ? YES : NO;
                }
            }
        }

        NSData *data = [NSData dataWithBytesNoCopy:const_cast<void *>(processed.data())
                                            length:processed.size()
                                      freeWhenDone:NO];
        ModelHandle *modelHandle = [model_manager_ loadModelFromAOTData:data
                                                          configuration:configuration
                                                             methodName:methodNameStr
                                                           functionName:functionNameStr
                                                              cachePath:cachePath
                                                            useNewCache:useNewCache
                                                                  error:&localError];
        if (localError != nil) {
            ETCoreMLLogError(localError, "Model init failed");
        }
        return modelHandle;
    }

    bool execute(Handle* handle,
                 std::vector<MultiArray>& args,
                 const ModelLoggingOptions& logging_options,
                 ModelEventLogger *event_logger,
                 std::error_code& ec) const noexcept override {
        NSError *localError = nil;
        if (![model_manager_ executeModelWithHandle:handle
                                            argsVec:args
                                     loggingOptions:logging_options
                                        eventLogger:event_logger
                                              error:&localError]) {
            if (localError != nil) {
                ETCoreMLLogError(localError, "Model execution failed");
                ec = static_cast<ErrorCode>(localError.code);
            }
            return false;
        }

        return true;
    }

    bool is_valid_handle(Handle* handle) const noexcept override {
        return [model_manager_ modelWithHandle:handle] != nil;
    }

    bool is_available() const noexcept override {
        return static_cast<bool>(model_manager_.isAvailable);
    }

    std::pair<size_t, size_t> get_num_arguments(Handle* handle) const noexcept override {
        ETCoreMLModel *model = [model_manager_ modelWithHandle:handle];
        return {model.orderedInputNames.count, model.orderedOutputNames.count};
    }

    void destroy(Handle* handle) const noexcept override {
        [model_manager_ unloadModelWithHandle:handle];
    }

    bool purge_models_cache() const noexcept override {
        NSError *localError = nil;
        bool result = static_cast<bool>([model_manager_ purgeModelsCacheAndReturnError:&localError]);
        return result;
    }

    ETCoreMLModelManagerDelegate *model_manager_;
    Config config_;
};

std::shared_ptr<BackendDelegate> BackendDelegate::make(const Config& config) {
    return std::make_shared<BackendDelegateImpl>(config);
}
} //namespace executorchcoreml
