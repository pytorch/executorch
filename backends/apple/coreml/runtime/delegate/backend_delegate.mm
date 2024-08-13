//
// backend_delegate.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.


#import <ETCoreMLAssetManager.h>
#import <ETCoreMLModel.h>
#import <ETCoreMLModelManager.h>
#import <ETCoreMLStrings.h>
#import <backend_delegate.h>
#import <model_event_logger.h>
#import <multiarray.h>

namespace  {
using namespace executorchcoreml;

MLComputeUnits get_compute_units(const Buffer& buffer) {
    std::string value(reinterpret_cast<const char *>(buffer.data()), buffer.size());
    if (value == std::string(ETCoreMLStrings.cpuComputeUnitName.UTF8String)) {
        return MLComputeUnitsCPUOnly;
    } else if (value == std::string(ETCoreMLStrings.cpuAndGpuComputeUnitsName.UTF8String)) {
        return MLComputeUnitsCPUAndGPU;
    } else if (value == std::string(ETCoreMLStrings.cpuAndNeuralEngineComputeUnitsName.UTF8String)) {
        return MLComputeUnitsCPUAndNeuralEngine;
    } else {
        return MLComputeUnitsAll;
    }
}

MLModelConfiguration *get_model_configuration(const std::unordered_map<std::string, Buffer>& specs) {
    std::string key_name(ETCoreMLStrings.computeUnitsKeyName.UTF8String);
    MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
    for (const auto& [key, buffer] : specs) {
        if (key == key_name) {
            configuration.computeUnits = get_compute_units(buffer);
            break;
        }
    }
    
    return configuration;
}

NSURL * _Nullable create_directory_if_needed(NSURL *url,
                                             NSFileManager *fileManager,
                                             NSError * __autoreleasing *error) {
    if (![fileManager fileExistsAtPath:url.path] &&
        ![fileManager createDirectoryAtURL:url withIntermediateDirectories:YES attributes:@{} error:error]) {
        return nil;
    }
    
    return url;
}

ETCoreMLAssetManager * _Nullable create_asset_manager(NSString *assets_directory_path,
                                                      NSString *trash_directory_path,
                                                      NSString *database_directory_path,
                                                      NSString *database_name,
                                                      NSInteger max_assets_size_in_bytes,
                                                      NSError * __autoreleasing *error) {
    NSFileManager *fm  = [[NSFileManager alloc] init];
    
    NSURL *assets_directory_url = [NSURL fileURLWithPath:assets_directory_path];
    if (!create_directory_if_needed(assets_directory_url, fm, error)) {
        return nil;
    }
    
    NSURL *trash_directory_url = [NSURL fileURLWithPath:trash_directory_path];
    if (!create_directory_if_needed(trash_directory_url, fm, error)) {
        return nil;
    }
    
    NSURL *database_directory_url = [NSURL fileURLWithPath:database_directory_path];
    if (!create_directory_if_needed(database_directory_url, fm, error)) {
        return nil;
    }
    
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
                               error:(NSError* __autoreleasing*)error;

- (BOOL)executeModelWithHandle:(ModelHandle*)handle
                       argsVec:(const std::vector<executorchcoreml::MultiArray>&)argsVec
                loggingOptions:(const executorchcoreml::ModelLoggingOptions&)loggingOptions
                   eventLogger:(const executorchcoreml::ModelEventLogger* _Nullable)eventLogger
                         error:(NSError* __autoreleasing*)error;

- (BOOL)unloadModelWithHandle:(ModelHandle*)handle;

- (BOOL)purgeModelsCacheAndReturnError:(NSError * _Nullable __autoreleasing *)error;

@property (assign, readonly, nonatomic) BackendDelegate::Config config;
@property (strong, readonly, nonatomic) dispatch_queue_t syncQueue;
@property (strong, nonatomic, nullable) ETCoreMLModelManager *impl;
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
    if (![self loadAndReturnError:error]) {
        return nil;
    }
    
    auto handle = [self.impl loadModelFromAOTData:data
                                    configuration:configuration
                                            error:error];
    if ((handle != NULL) && self.config.should_prewarm_model) {
        [self.impl prewarmModelWithHandle:handle error:nil];
    }

    return handle;
}

- (BOOL)executeModelWithHandle:(ModelHandle*)handle
                       argsVec:(const std::vector<executorchcoreml::MultiArray>&)argsVec
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
    
    Handle *init(Buffer processed,const std::unordered_map<std::string, Buffer>& specs) const noexcept override {
        NSError *localError = nil;
        MLModelConfiguration *configuration = get_model_configuration(specs);
        NSData *data = [NSData dataWithBytesNoCopy:const_cast<void *>(processed.data())
                                            length:processed.size()
                                      freeWhenDone:NO];
        ModelHandle *modelHandle = [model_manager_ loadModelFromAOTData:data
                                                          configuration:configuration
                                                                  error:&localError];
        return modelHandle;
    }
    
    bool execute(Handle* handle,
                 const std::vector<MultiArray>& args,
                 const ModelLoggingOptions& logging_options,
                 ModelEventLogger *event_logger,
                 std::error_code& ec) const noexcept override {
        NSError *error = nil;
        if (![model_manager_ executeModelWithHandle:handle
                                            argsVec:args
                                     loggingOptions:logging_options
                                        eventLogger:event_logger
                                              error:&error]) {
            ec = static_cast<ErrorCode>(error.code);
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
