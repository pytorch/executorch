//
// backend_delegate.hpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#include <model_logging_options.h>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace executorchcoreml {

class ModelEventLogger;
class MultiArray;
class Buffer;

/// An abstract class for a CoreML delegate to implement.
class BackendDelegate {
public:
    /// The model handle.
    using Handle = void;

    struct Config {
        // Max models cache size in bytes.
        size_t max_models_cache_size = 10 * size_t(1024) * size_t(1024) * size_t(1024);
        // If set to `true`, delegate pre-warms the most recently used asset.
        bool should_prewarm_asset = true;
        // If set to `true`, delegate pre-warms the model in `init`.
        bool should_prewarm_model = true;
    };

    /// The error codes for the `BackendDelegate`.
    enum class ErrorCode : int8_t {
        CorruptedData = 1, // AOT blob can't be parsed.
        CorruptedMetadata, // AOT blob has incorrect or missing metadata.
        CorruptedModel, // AOT blob has incorrect or missing CoreML model.
        BrokenModel, // Model doesn't match the input and output specifications.
        CompilationFailed, // Model failed to compile.
        ModelSaveFailed, // Failed to save Model to disk.
        ModelCacheCreationFailed // Failed to create models cache.
    };

    /// The error category for `BackendDelegate` errors.
    struct ErrorCategory final : public std::error_category {
    public:
        /// Returns the name of the category.
        inline const char* name() const noexcept override { return "CoreMLBackend"; }

        /// Returns a message from the error code.
        std::string message(int code) const override;
    };

    inline BackendDelegate() noexcept { }
    virtual ~BackendDelegate() noexcept = default;

    BackendDelegate(BackendDelegate const&) = delete;
    BackendDelegate& operator=(BackendDelegate const&) = delete;

    BackendDelegate(BackendDelegate&&) = default;
    BackendDelegate& operator=(BackendDelegate&&) = default;

    /// Must initialize a CoreML model.
    ///
    /// The method receives the AOT blob that's embedded in the executorch
    /// Program. The implementation must initialize the model and prepare it for
    /// execution.
    ///
    /// @param processed The AOT blob.
    /// @param specs The specs at the time of compilation.
    /// @retval An opaque handle to the initialized blob or `nullptr` if the
    /// initialization failed.
    virtual Handle* init(Buffer processed, const std::unordered_map<std::string, Buffer>& specs) const noexcept = 0;

    /// Must execute the CoreML model with the specified handle.
    ///
    /// The `args` are inputs and outputs combined. It's the responsibility of the
    /// implementation to find the inputs and the outputs from `args`. The
    /// implementation must execute the model with the inputs and must populate
    /// the outputs from the model prediction outputs.
    ///
    /// @param handle The model handle.
    /// @param args The inputs and outputs to the model.
    /// @param logging_options The model logging options.
    /// @param event_logger The model event logger.
    /// @param error   On failure, error is filled with the failure information.
    /// @retval `true` if the execution succeeded otherwise `false`.
    virtual bool execute(Handle* handle,
                         const std::vector<MultiArray>& args,
                         const ModelLoggingOptions& logging_options,
                         ModelEventLogger* event_logger,
                         std::error_code& error) const noexcept = 0;

    /// Must return `true` if the delegate is available for execution otherwise
    /// `false`.
    virtual bool is_available() const noexcept = 0;

    /// Must returns the number of inputs and the number of outputs for the
    /// specified handle.
    ///
    /// The returned pair's first value is the number of inputs and the second
    /// value is the number of outputs.
    ///
    /// @param handle The model handle.
    /// @retval A pair with the number of inputs and the number of outputs.
    virtual std::pair<size_t, size_t> get_num_arguments(Handle* handle) const noexcept = 0;

    /// Checks if the model handle is valid.
    ///
    /// @param handle The model handle.
    /// @retval `true` if the model handle is valid otherwise `false`.
    virtual bool is_valid_handle(Handle* handle) const noexcept = 0;

    /// Must unload the CoreML model with the specified handle.
    ///
    /// The returned pair's first value is the number of inputs and the second
    /// value is the number of outputs.
    ///
    /// @param handle The model handle.
    virtual void destroy(Handle* handle) const noexcept = 0;

    /// Purges the models cache.
    ///
    /// Compiled models are stored on-disk to improve the model load time. The
    /// method tries to remove all the models that are not currently in-use.
    virtual bool purge_models_cache() const noexcept = 0;

    /// Returns a delegate implementation with the specified config.
    ///
    /// @param config The delegate config.
    /// @retval A delegate implementation.
    static std::shared_ptr<BackendDelegate> make(const Config& config);
};

/// Constructs a `std::error_code` from  a`BackendDelegate::ErrorCode`.
///
/// @param code The backend error code.
/// @retval A `std::error_code` constructed from
/// the`BackendDelegate::ErrorCode`.
inline std::error_code make_error_code(BackendDelegate::ErrorCode code) {
    static BackendDelegate::ErrorCategory errorCategory;
    return { static_cast<int>(code), errorCategory };
}
} // namespace executorchcoreml

namespace std {
template <> struct is_error_code_enum<executorchcoreml::BackendDelegate::ErrorCode> : true_type { };
} // namespace std
