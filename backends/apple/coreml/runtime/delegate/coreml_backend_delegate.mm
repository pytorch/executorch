//
// coreml_backend_delegate.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <ETCoreMLLogging.h>
#import <ETCoreMLModel.h>
#import <ETCoreMLStrings.h>
#import <backend_delegate.h>
#import <coreml_backend/delegate.h>
#import <executorch/runtime/core/evalue.h>
#import <executorch/runtime/platform/log.h>
#import <memory>
#import <model_event_logger.h>
#import <model_logging_options.h>
#import <multiarray.h>
#import <objc_safe_cast.h>
#import <unordered_map>
#import <vector>

#ifdef ET_EVENT_TRACER_ENABLED
#import <model_event_logger_impl.h>
#endif

namespace {
using namespace torch::executor;
using namespace executorchcoreml;

std::optional<MultiArray::DataType> get_data_type(ScalarType scalar_type) {
    switch (scalar_type) {
        case ScalarType::Bool:
            return MultiArray::DataType::Bool;
        case ScalarType::Byte:
            return MultiArray::DataType::Byte;
        case ScalarType::Short:
            return MultiArray::DataType::Short;
        case ScalarType::Int:
            return MultiArray::DataType::Int32;
        case ScalarType::Long:
            return MultiArray::DataType::Int64;
        case ScalarType::Half:
            return MultiArray::DataType::Float16;
        case ScalarType::Float:
            return MultiArray::DataType::Float32;
        case ScalarType::Double:
            return MultiArray::DataType::Float64;
        default:
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
        ET_LOG(Error, "%s: DataType=%d is not supported", ETCoreMLStrings.delegateIdentifier.UTF8String, (int)tensor.scalar_type());
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
            config.should_prewarm_model = static_cast<bool>(should_prewarm_model.boolValue);
        }
    }
    
    {
        NSNumber *should_prewarm_asset = SAFE_CAST(dict[@"shouldPrewarmAsset"], NSNumber);
        if (should_prewarm_asset) {
            config.should_prewarm_asset = static_cast<bool>(should_prewarm_asset.boolValue);
        }
    }
    
    {
        NSNumber *max_models_cache_size_in_bytes = SAFE_CAST(dict[@"maxModelsCacheSizeInBytes"], NSNumber);
        if (max_models_cache_size_in_bytes) {
            config.max_models_cache_size = max_models_cache_size_in_bytes.unsignedLongLongValue;
        }
    }
    
    return config;
}

BackendDelegate::Config get_delegate_config(NSString *config_name) {
    NSURL *config_url = [NSBundle.mainBundle URLForResource:config_name withExtension:@"plist"];
    config_url = config_url ?: [[NSBundle bundleForClass:ETCoreMLModel.class] URLForResource:config_name withExtension:@"plist"];
    auto config = parse_config(config_url);
    return config.has_value() ? config.value() : BackendDelegate::Config();
}

ModelLoggingOptions get_logging_options(BackendExecutionContext& context) {
    ModelLoggingOptions options;
    auto event_tracer = context.event_tracer();
    if (event_tracer) {
        options.log_profiling_info = true;
        auto debug_level = event_tracer->event_tracer_debug_level();
        options.log_intermediate_tensors = (debug_level >= EventTracerDebugLogLevel::kIntermediateOutputs);
    }
    
    return options;
}

} //namespace

namespace torch {
namespace executor {

using namespace executorchcoreml;

CoreMLBackendDelegate::CoreMLBackendDelegate() noexcept
:impl_(BackendDelegate::make(get_delegate_config(ETCoreMLStrings.configPlistName)))
{}

Result<DelegateHandle *>
CoreMLBackendDelegate::init(BackendInitContext& context,
                            FreeableBuffer* processed,
                            ArrayRef<CompileSpec> specs) const {
    ET_LOG(Debug, "%s: init called.", ETCoreMLStrings.delegateIdentifier.UTF8String);
    std::unordered_map<std::string, Buffer> specs_map;
    specs_map.reserve(specs.size());
    for (auto it = specs.cbegin(); it != specs.cend(); ++it) {
        auto& spec = *(it);
        auto buffer = Buffer(spec.value.buffer, spec.value.nbytes);
        specs_map.emplace(spec.key, std::move(buffer));
    }
    
    auto buffer = Buffer(processed->data(), processed->size());
    std::error_code error;
    auto handle = impl_->init(std::move(buffer), specs_map);
    ET_CHECK_OR_RETURN_ERROR(handle != nullptr,
                             InvalidProgram,
                             "%s: Failed to init the model.", ETCoreMLStrings.delegateIdentifier.UTF8String);
    processed->Free();
    return handle;
}

Error CoreMLBackendDelegate::execute(BackendExecutionContext& context,
                                     DelegateHandle* handle,
                                     EValue** args) const {
    const auto& nArgs = impl_->get_num_arguments(handle);
    std::vector<MultiArray> delegate_args;
    size_t nInputs = nArgs.first;
    size_t nOutputs = nArgs.second;
    delegate_args.reserve(nInputs + nOutputs);
    
    // inputs
    for (size_t i = 0; i < nInputs; i++) {
        auto multi_array = get_multi_array(args[i], ArgType::Input);
        ET_CHECK_OR_RETURN_ERROR(multi_array.has_value(),
                                 Internal,
                                 "%s: Failed to create multiarray from input at args[%zu]", ETCoreMLStrings.delegateIdentifier.UTF8String, i);
        delegate_args.emplace_back(std::move(multi_array.value()));
    }
    
    // outputs
    for (size_t i = nInputs; i < nInputs + nOutputs; i++) {
        auto multi_array = get_multi_array(args[i], ArgType::Output);
        ET_CHECK_OR_RETURN_ERROR(multi_array.has_value(),
                                 Internal,
                                 "%s: Failed to create multiarray from output at args[%zu]", ETCoreMLStrings.delegateIdentifier.UTF8String, i);
        delegate_args.emplace_back(std::move(multi_array.value()));
    }
    
    auto logging_options = get_logging_options(context);
    std::error_code ec;
#ifdef ET_EVENT_TRACER_ENABLED
    auto event_logger = ModelEventLoggerImpl(context.event_tracer());
    ET_CHECK_OR_RETURN_ERROR(impl_->execute(handle, delegate_args, logging_options, &event_logger, ec),
                             DelegateInvalidHandle,
                             "%s: Failed to run the model.",
                             ETCoreMLStrings.delegateIdentifier.UTF8String);
#else
    ET_CHECK_OR_RETURN_ERROR(impl_->execute(handle, delegate_args, logging_options, nullptr, ec),
                             DelegateInvalidHandle,
                             "%s: Failed to run the model.",
                             ETCoreMLStrings.delegateIdentifier.UTF8String);
#endif
    
    return Error::Ok;
}

bool CoreMLBackendDelegate::is_available() const {
    ET_LOG(Debug, "%s: is_available called.", ETCoreMLStrings.delegateIdentifier.UTF8String);
    return impl_->is_available();
}

void CoreMLBackendDelegate::destroy(DelegateHandle* handle) const {
    ET_LOG(Debug, "%s: destroy called.", ETCoreMLStrings.delegateIdentifier.UTF8String);
    impl_->destroy(handle);
}

bool CoreMLBackendDelegate::purge_models_cache() const noexcept {
    ET_LOG(Debug, "%s: purge_models_cache called.", ETCoreMLStrings.delegateIdentifier.UTF8String);
    return impl_->purge_models_cache();
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
