//
// delegate.h
//
// Copyright © 2023 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

#include <memory>

namespace executorchcoreml {
class BackendDelegate;
}

namespace torch {
namespace executor {

class CoreMLBackendDelegate final : public ::executorch::runtime::BackendInterface {
public:
    CoreMLBackendDelegate() noexcept;
    ~CoreMLBackendDelegate() = default;

    /// Loads a CoreML model from AOT blob.
    ///
    /// @param context An init context specific to the CoreML backend.
    /// @param processed The AOT blob that was produced by a call to CoreML's
    /// backend `preprocess` method.
    /// @param compileSpecs The exact same compiler specification that was used to
    /// produce `processed`.
    /// @retval On success, an opaque handle representing the loaded model
    /// otherwise  an`Error` case.
    Result<DelegateHandle*>
    init(BackendInitContext& context, FreeableBuffer* processed, ArrayRef<CompileSpec> compileSpecs) const override;

    /// Executes the loaded model.
    ///
    /// @param context An execution context specific to the CoreML backend.
    /// @param handle The handle returned by an earlier call to `init`.
    /// @param args The models inputs and outputs.
    /// @retval On success, `Error::Ok` otherwise any other `Error` case.
    Error execute(BackendExecutionContext& context, DelegateHandle* handle, EValue** args) const override;

    /// Returns `true` if the delegate is available otherwise `false`.
    bool is_available() const override;

    /// Unloads the loaded CoreML model with the  specified handle.
    ///
    /// @param handle The handle returned by an earlier call to `init`.
    void destroy(DelegateHandle* handle) const override;

    /// Returns the registered `CoreMLBackendDelegate` instance.
    static CoreMLBackendDelegate* get_registered_delegate() noexcept;

    /// Purges the models cache.
    ///
    /// The cached models are moved to a temporary directory and are
    /// asynchronously deleted.
    bool purge_models_cache() const noexcept;

private:
    std::shared_ptr<executorchcoreml::BackendDelegate> impl_;
};
} // namespace executor
} // namespace torch
