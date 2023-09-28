//
// coreml_backend_delegate.mm
//
// Copyright © 2023 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "coreml_backend_delegate.h"

#import <memory>
#import <unordered_map>
#import <vector>

#import <multiarray.h>

#import <ETCoreMLLogging.h>
#import <ETCoreMLModel.h>
#import <ETCoreMLStrings.h>

#import <executorch/runtime/core/evalue.h>

namespace {
using namespace torch::executor;
using namespace executorchcoreml;

inline id check_class(id obj, Class cls) {
    return [obj isKindOfClass:cls] ? obj : nil;
}

#define SAFE_CAST(Object, Type) ((Type *)check_class(Object, [Type class]))

std::optional<MultiArray::DataType> get_data_type(ScalarType scalarType) {
    if (scalarType == ScalarType::Float) {
        return MultiArray::DataType::Float;
    } else if (scalarType == ScalarType::Double) {
        return MultiArray::DataType::Double;
    } else if (scalarType == ScalarType::Half) {
        return MultiArray::DataType::Float16;
    } else if (scalarType == ScalarType::Int) {
        return MultiArray::DataType::Int;
    } else {
        return std::nullopt;
    }
}

enum class ArgType: uint8_t {
    Input,
    Output
};

std::optional<MultiArray> get_multi_array(EValue *eValue, ArgType argType) {
    if (!eValue->isTensor()) {
        return std::nullopt;
    }
    
    auto tensor = eValue->toTensor();
    auto dataType = get_data_type(tensor.scalar_type());
    if (!dataType.has_value()) {
        return std::nullopt;
    }
    
    std::vector<ssize_t> strides(tensor.strides().begin(), tensor.strides().end());
    std::vector<size_t> shape(tensor.sizes().begin(), tensor.sizes().end());
    MultiArray::MemoryLayout layout(dataType.value(), std::move(shape), std::move(strides));
    switch (argType) {
        case ArgType::Input: {
            return MultiArray(const_cast<void *>(tensor.const_data_ptr()), layout);
        }
        case ArgType::Output: {
            return MultiArray(tensor.mutable_data_ptr(), layout);
        }
    }
}

std::optional<BackendDelegate::Config> parse_config(NSURL *plistURL) {
    NSDictionary<NSString *, id> *dict = [NSDictionary dictionaryWithContentsOfURL:plistURL];
    if (!dict) {
        return std::nullopt;
    }
    
    BackendDelegate::Config config;
    {
        NSNumber *should_prewarm_model = SAFE_CAST(dict[@"shouldPrewarmModel"], NSNumber);
        if (should_prewarm_model) {
            config.shouldPrewarmModel = static_cast<bool>(should_prewarm_model.boolValue);
        }
    }
    
    {
        NSNumber *should_prewarm_asset = SAFE_CAST(dict[@"shouldPrewarmAsset"], NSNumber);
        if (should_prewarm_asset) {
            config.shouldPrewarmAsset = static_cast<bool>(should_prewarm_asset.boolValue);
        }
    }
    
    {
        NSNumber *max_models_cache_size_in_bytes = SAFE_CAST(dict[@"maxModelsCacheSizeInBytes"], NSNumber);
        if (max_models_cache_size_in_bytes) {
            config.maxModelsCacheSizeInBytes = max_models_cache_size_in_bytes.unsignedLongLongValue;
        }
    }
    
    return config;
}

BackendDelegate::Config get_delegate_config(NSString *config_name) {
    NSURL *config_url = [[NSBundle mainBundle] URLForResource:config_name withExtension:@"plist"];
    config_url = config_url ?: [[NSBundle bundleForClass:ETCoreMLModel.class] URLForResource:config_name withExtension:@"plist"];
    auto config = parse_config(config_url);
    return config.has_value() ? config.value() : BackendDelegate::Config();
}
} //namespace

namespace torch {
namespace executor {

using namespace executorchcoreml;

CoreMLBackendDelegate::CoreMLBackendDelegate() noexcept
:delegate_(BackendDelegate::make(get_delegate_config(ETCoreMLStrings.configPlistName)))
{}

Result<DelegateHandle *>
CoreMLBackendDelegate::init(BackendInitContext& context,
                            FreeableBuffer* processed,
                            ArrayRef<CompileSpec> specs) const {
    ET_LOG(Debug, "%s: init called.", ETCoreMLStrings.delegateIdentifier.UTF8String);
    std::unordered_map<std::string, Buffer> specsMap;
    specsMap.reserve(specs.size());
    for (auto it = specs.cbegin(); it != specs.cend(); ++it) {
        auto& spec = *(it);
        auto buffer = Buffer(spec.value.buffer, spec.value.nbytes);
        specsMap.emplace(spec.key, std::move(buffer));
    }
    
    auto buffer = Buffer(processed->data(), processed->size());
    std::error_code error;
    auto handle = delegate_->init(std::move(buffer), specsMap);
    ET_CHECK_OR_RETURN_ERROR(handle != nullptr,
                             InvalidProgram,
                             "%s: Failed to init the model.", ETCoreMLStrings.delegateIdentifier.UTF8String);
    if (handle) {
        processed->Free();
    }
    
    return handle;
}

Error CoreMLBackendDelegate::execute(BackendExecutionContext& context,
                                     DelegateHandle* handle,
                                     EValue** args) const {
    const auto& nArgs = delegate_->get_num_arguments(handle);
    std::vector<MultiArray> delegateArgs;
    size_t nInputs = nArgs.first;
    size_t nOutputs = nArgs.second;
    delegateArgs.reserve(nInputs + nOutputs);
    
    // inputs
    for (size_t i = 0; i < nInputs; i++) {
        auto multi_array = get_multi_array(args[i], ArgType::Input);
        ET_CHECK_OR_RETURN_ERROR(multi_array.has_value(),
                                 Internal,
                                 "%s: Expected tensor at args[%zu]", ETCoreMLStrings.delegateIdentifier.UTF8String, i);
        delegateArgs.emplace_back(std::move(multi_array.value()));
    }
    
    // outputs
    for (size_t i = nInputs; i < nInputs + nOutputs; i++) {
        auto multi_array = get_multi_array(args[i], ArgType::Output);
        ET_CHECK_OR_RETURN_ERROR(multi_array.has_value(),
                                 Internal,
                                 "%s: Expected tensor at args[%zu]", ETCoreMLStrings.delegateIdentifier.UTF8String, i);
        delegateArgs.emplace_back(std::move(multi_array.value()));
    }
    
    std::error_code error;
    ET_CHECK_OR_RETURN_ERROR(delegate_->execute(handle, delegateArgs, error),
                             DelegateInvalidHandle,
                             "%s: Failed to run the model.",
                             ETCoreMLStrings.delegateIdentifier.UTF8String);
    return Error::Ok;
}

bool CoreMLBackendDelegate::is_available() const {
    ET_LOG(Debug, "%s: is_available called.", ETCoreMLStrings.delegateIdentifier.UTF8String);
    return delegate_->is_available();
}

void CoreMLBackendDelegate::destroy(DelegateHandle* handle) const {
    ET_LOG(Debug, "%s: destroy called.", ETCoreMLStrings.delegateIdentifier.UTF8String);
    delegate_->destroy(handle);
}

bool CoreMLBackendDelegate::purge_models_cache() const noexcept {
    ET_LOG(Debug, "%s: purge_models_cache called.", ETCoreMLStrings.delegateIdentifier.UTF8String);
    return delegate_->purge_models_cache();
}

CoreMLBackendDelegate *CoreMLBackendDelegate::get_registered_delegate() noexcept {
    return static_cast<CoreMLBackendDelegate *>(get_backend_class(ETCoreMLStrings.delegateIdentifier.UTF8String));
}

namespace {
auto cls = CoreMLBackendDelegate();
Backend backend{ETCoreMLStrings.delegateIdentifier.UTF8String, &cls};
static auto success_with_compiler = register_backend(backend);
}

} // namespace executor
} // namespace torch

