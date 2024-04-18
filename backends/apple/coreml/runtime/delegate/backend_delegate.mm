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
#import <atomic>
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
    :BackendDelegate(), config_(config) {
        NSError *localError = nil;
        ETCoreMLAssetManager *asset_manager = create_asset_manager(ETCoreMLStrings.assetsDirectoryPath,
                                                                   ETCoreMLStrings.trashDirectoryPath,
                                                                   ETCoreMLStrings.databaseDirectoryPath,
                                                                   ETCoreMLStrings.databaseName,
                                                                   config.max_models_cache_size,
                                                                   &localError);
        
        model_manager_ = (asset_manager != nil) ? [[ETCoreMLModelManager alloc] initWithAssetManager:asset_manager] : nil;
        if (model_manager_ != nil && config_.should_prewarm_asset) {
            [model_manager_ prewarmRecentlyUsedAssetsWithMaxCount:1];
        }
        available_.store(model_manager_ != nil, std::memory_order_seq_cst);
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
        if (modelHandle && config_.should_prewarm_model) {
            NSError *localError = nil;
            [model_manager_ prewarmModelWithHandle:modelHandle error:&localError];
        }
        
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
        return available_.load(std::memory_order_acquire);
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
        bool result = static_cast<bool>([model_manager_.assetManager purge:&localError]);
        return result;
    }
    
    ETCoreMLModelManager *model_manager_;
    std::atomic<bool> available_;
    Config config_;
};

std::shared_ptr<BackendDelegate> BackendDelegate::make(const Config& config) {
    return std::make_shared<BackendDelegateImpl>(config);
}
} //namespace executorchcoreml

