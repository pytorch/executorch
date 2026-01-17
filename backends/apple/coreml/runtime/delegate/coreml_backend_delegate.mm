//
// coreml_backend_delegate.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "coreml_backend/delegate.h"

#import "backend_delegate.h"
#import "ETCoreMLLogging.h"
#import "ETCoreMLModel.h"
#import "ETCoreMLStrings.h"
#import "model_event_logger.h"
#import "model_logging_options.h"
#import "multiarray.h"
#import "objc_safe_cast.h"

#import <executorch/runtime/core/evalue.h>
#import <executorch/runtime/platform/log.h>
#import <executorch/runtime/kernel/kernel_includes.h>

#include <array>
#import <memory>
#import <unordered_map>
#import <vector>

#import <Foundation/Foundation.h>

#ifdef ET_EVENT_TRACER_ENABLED
#import <model_event_logger_impl.h>
#endif

namespace {
using namespace executorchcoreml;

using executorch::aten::ScalarType;
using executorch::runtime::ArrayRef;
using executorch::runtime::Backend;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::EValue;
using executorch::runtime::Error;
using executorch::runtime::EventTracerDebugLogLevel;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::get_backend_class;
using executorch::runtime::Result;
using executorch::aten::SizesType;
using executorch::runtime::Span;
using executorch::aten::Tensor;
using executorch::runtime::kTensorDimensionLimit;

/// Checks if the processed bytes represent a JSON reference to NamedDataStore.
/// The JSON format is: {"version": 1, "key": "...", "method": "..."}
///
/// @param data Pointer to the processed bytes.
/// @param size Size of the processed bytes.
/// @return true if the bytes appear to be a JSON reference, false otherwise.
bool isNamedDataReference(const void* data, size_t size) {
    // Quick check: JSON starts with '{' and should be small (< 512 bytes)
    if (size < 2 || size > 512 || static_cast<const char*>(data)[0] != '{') {
        return false;
    }
    
    // Try to parse as JSON using Foundation
    NSData *jsonData = [NSData dataWithBytesNoCopy:const_cast<void*>(data)
                                            length:size
                                      freeWhenDone:NO];
    NSError *error = nil;
    id jsonObject = [NSJSONSerialization JSONObjectWithData:jsonData
                                                    options:0
                                                      error:&error];
    if (error != nil || ![jsonObject isKindOfClass:[NSDictionary class]]) {
        return false;
    }
    
    NSDictionary *dict = (NSDictionary *)jsonObject;
    // Check for required fields: "version" and "key"
    return dict[@"version"] != nil && [dict[@"key"] isKindOfClass:[NSString class]];
}

/// Parses the JSON reference and extracts the NamedDataStore key.
/// Expected format: {"version": 1, "key": "...", "method": "..."}
///
/// @param data Pointer to the JSON bytes.
/// @param size Size of the JSON bytes.
/// @return The extracted key, or empty string if parsing fails.
std::string parseNamedDataKey(const void* data, size_t size) {
    NSData *jsonData = [NSData dataWithBytesNoCopy:const_cast<void*>(data)
                                            length:size
                                      freeWhenDone:NO];
    NSError *error = nil;
    id jsonObject = [NSJSONSerialization JSONObjectWithData:jsonData
                                                    options:0
                                                      error:&error];
    if (error != nil || ![jsonObject isKindOfClass:[NSDictionary class]]) {
        ET_LOG(Error, "Failed to parse JSON reference");
        return "";
    }
    
    NSDictionary *dict = (NSDictionary *)jsonObject;
    NSString *key = dict[@"key"];
    if ([key isKindOfClass:[NSString class]]) {
        return std::string([key UTF8String]);
    }
    
    return "";
}

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

    // If tensor is rank 0, wrap in rank 1
    // See https://github.com/apple/coremltools/blob/8.2/coremltools/converters/mil/frontend/torch/exir_utils.py#L73
    if (shape.size() == 0) {
        shape.push_back(1);
        strides.push_back(1);
    }

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

namespace executorch {
namespace backends {
namespace coreml {

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

    Buffer buffer(nullptr, 0);  // Will be set below
    
    // Check if processed bytes is a JSON reference to NamedDataStore
    if (isNamedDataReference(processed->data(), processed->size())) {
        // Parse the key from the JSON reference
        std::string key = parseNamedDataKey(processed->data(), processed->size());
        ET_CHECK_OR_RETURN_ERROR(!key.empty(),
                                 InvalidProgram,
                                 "%s: Failed to parse NamedDataStore key from JSON reference.",
                                 ETCoreMLStrings.delegateIdentifier.UTF8String);
        
        ET_LOG(Debug, "%s: Loading model from NamedDataStore with key: %s",
               ETCoreMLStrings.delegateIdentifier.UTF8String, key.c_str());
        
        // Get the NamedDataMap from context
        const auto* named_data_map = context.get_named_data_map();
        ET_CHECK_OR_RETURN_ERROR(named_data_map != nullptr,
                                 InvalidProgram,
                                 "%s: NamedDataMap is null but processed bytes is a JSON reference.",
                                 ETCoreMLStrings.delegateIdentifier.UTF8String);
        
        // Load the model data from NamedDataMap
        auto result = named_data_map->get_data(key.c_str());
        ET_CHECK_OR_RETURN_ERROR(result.ok(),
                                 InvalidProgram,
                                 "%s: Failed to load model data from NamedDataStore with key: %s",
                                 ETCoreMLStrings.delegateIdentifier.UTF8String, key.c_str());
        
        // Move the result into the incoming FreeableBuffer so its lifetime matches `processed`
        processed->~FreeableBuffer();
        new (processed) FreeableBuffer(std::move(result.get()));
        buffer = Buffer(processed->data(), processed->size());
        
        ET_LOG(Debug, "%s: Loaded %zu bytes from NamedDataStore",
               ETCoreMLStrings.delegateIdentifier.UTF8String, processed->size());
    } else {
        // Legacy path: use processed bytes directly
        buffer = Buffer(processed->data(), processed->size());
    }

    // Get method name for multifunction model support
    const char* method_name = context.get_method_name();
    if (method_name != nullptr) {
        ET_LOG(Debug, "%s: Method name: %s", ETCoreMLStrings.delegateIdentifier.UTF8String, method_name);
    }

    std::error_code error;
    auto handle = impl_->init(std::move(buffer), specs_map, method_name);
    ET_CHECK_OR_RETURN_ERROR(handle != nullptr,
                             InvalidProgram,
                             "%s: Failed to init the model.", ETCoreMLStrings.delegateIdentifier.UTF8String);
    processed->Free();
    return handle;
}

Error CoreMLBackendDelegate::execute(BackendExecutionContext& context,
                                     DelegateHandle* handle,
                                     Span<EValue*> args) const {
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

    // Resize for dynamic shape
    std::array<SizesType, kTensorDimensionLimit> new_shape;
    for (size_t i = nInputs; i < nInputs + nOutputs; i++) {
        Tensor& t = args[i]->toTensor();
        // If t has rank 0, do not resize.  delegate_args[i] will have rank 1
        // because we resized it in get_multi_array
        if (t.dim() == 0) {
            continue;
        }

        int rank = delegate_args[i].layout().rank();
        assert (rank <= new_shape.size());
        for (int d = 0; d < rank; d++) {
            new_shape[d] = delegate_args[i].layout().shape()[d];
        }
        ET_CHECK_OR_RETURN_ERROR(
            resize_tensor(t, ArrayRef(new_shape.data(), rank)) == Error::Ok,
            DelegateInvalidHandle,
            "%s: Failed to resize delegate output %zu",  ETCoreMLStrings.delegateIdentifier.UTF8String, i);
    }

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
    #ifndef LAZY_LOAD_IOS_PYTORCH_INITIALIZER
        auto cls = CoreMLBackendDelegate();
        Backend backend{ETCoreMLStrings.delegateIdentifier.UTF8String, &cls};
        static auto success_with_compiler = register_backend(backend);
    #endif
}

} // namespace coreml
} // namespace backends
} // namespace executorch
