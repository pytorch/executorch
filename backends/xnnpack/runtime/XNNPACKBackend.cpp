/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/runtime/XNNCompiler.h>
#include <executorch/backends/xnnpack/runtime/XNNWeightsCache.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/executor/pte_data_map.h>

#include <memory>
#include <mutex>

#pragma clang diagnostic ignored "-Wglobal-constructors"

namespace executorch {
namespace backends {

using executorch::backends::xnnpack::delegate::XNNWeightsCache;
using executorch::ET_RUNTIME_NAMESPACE::Backend;
using executorch::ET_RUNTIME_NAMESPACE::BackendExecutionContext;
using executorch::ET_RUNTIME_NAMESPACE::BackendInitContext;
using executorch::ET_RUNTIME_NAMESPACE::CompileSpec;
using executorch::ET_RUNTIME_NAMESPACE::DelegateHandle;
using executorch::ET_RUNTIME_NAMESPACE::NamedDataMap;
using executorch::runtime::ArrayRef;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;
using executorch::runtime::Span;

class XnnpackBackend final
    : public ::executorch::ET_RUNTIME_NAMESPACE::BackendInterface {
 public:
  ~XnnpackBackend() = default;

  XnnpackBackend() {
    // Initialize XNNPACK
    xnn_status status = xnn_initialize(/*allocator=*/nullptr);
    if (status != xnn_status_success) {
      ET_LOG(
          Error,
          "Failed to initialize, XNNPACK status: 0x%x",
          (unsigned int)status);
      return;
    }

#ifdef ENABLE_XNNPACK_SHARED_WORKSPACE
    // Create a workspace for the XNNExecutor to use. This workspace will be
    // shared across all delegate instances.
    ET_LOG(Debug, "Creating XNN workspace");
    xnn_workspace_t workspace = nullptr;
    status = xnn_create_workspace(&workspace);
    if (status != xnn_status_success) {
      ET_LOG(
          Error,
          "Failed to create XNN workspace, XNNPACK status: 0x%x",
          (unsigned int)status);
      workspace = nullptr;
      return;
    }
    workspace_.reset(workspace);
    ET_LOG(Debug, "Created XNN workspace: %p", workspace_.get());
#endif // ENABLE_XNNPACK_SHARED_WORKSPACE
  }

  bool is_available() const override {
    return xnn_status_success == xnn_initialize(/*allocator=*/nullptr);
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
    auto executor = context.get_runtime_allocator()
                        ->allocateInstance<xnnpack::delegate::XNNExecutor>();
    if (executor == nullptr) {
      return Error::MemoryAllocationFailed;
    }

    const NamedDataMap* named_data_map = context.get_named_data_map();
    // thread safe. This can heppen when multiple threads call init() on
    // the same backend instance.
#ifdef ENABLE_XNNPACK_SHARED_WORKSPACE
    const std::lock_guard<std::mutex> lock(workspace_mutex_);
#endif

#ifdef ENABLE_XNNPACK_WEIGHTS_CACHE
    const std::lock_guard<std::mutex> lock_weight_cache(weights_cache_mutex_);
    weights_cache_->initialize_for_runtime(
        context.get_runtime_allocator(), named_data_map);
#endif

    // Executor has been allocated but not constructed, ensure that runtime_ is
    // nullptr by constructing it in place here. NOTE: Since we use placement
    // new and since this type is not trivially destructible, we must call the
    // destructor manually in destroy().
    new (executor) xnnpack::delegate::XNNExecutor;
    Error err = xnnpack::delegate::XNNCompiler::compileModel(
        processed->data(),
        processed->size(),
        executor,
        weights_cache_.get(),
        workspace_.get(),
        named_data_map);
    // This backend does not need its processed data after compiling the model.
    processed->Free();

    if (err != Error::Ok) {
      // destroy() won't be called on this handle, so we need to clean it up
      // now.
      executor->~XNNExecutor();

      ET_LOG(
          Error, "XNNCompiler::compileModel failed: 0x%x", (unsigned int)err);
      return err;
    }
    return executor;
  }

  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* handle,
      Span<EValue*> args) const override {
    auto executor = static_cast<xnnpack::delegate::XNNExecutor*>(handle);

#ifdef ENABLE_XNNPACK_SHARED_WORKSPACE
    const std::lock_guard<std::mutex> lock(workspace_mutex_);
#endif

#ifdef ENABLE_XNNPACK_WEIGHTS_CACHE
    const std::lock_guard<std::mutex> lock_weights_cache(weights_cache_mutex_);
#endif

    // Prepare Inputs/Outputs and Propagate Input Shapes
    Error err = executor->prepare_args(args);
    if (err != Error::Ok) {
      return err;
    }

    err = executor->forward(context);

    if (err != Error::Ok) {
      return err;
    }

    // Resize outputs and recast pointers if necessary
    err = executor->resize_outputs(args);

    return err;
  }

  void destroy(DelegateHandle* handle) const override {
    if (handle != nullptr) {
      // This is needed to serialize access to xnn_delete_runtime which is not
      // thread safe. This can heppen when multiple threads call destroy() on
      // the same backend instance.
#ifdef ENABLE_XNNPACK_SHARED_WORKSPACE
      const std::lock_guard<std::mutex> lock(workspace_mutex_);
#endif

      auto executor = static_cast<xnnpack::delegate::XNNExecutor*>(handle);

#ifdef ENABLE_XNNPACK_PROFILING
      executor->print_avg_op_timings();
#endif

#ifdef ENABLE_XNNPACK_WEIGHTS_CACHE
      const std::lock_guard<std::mutex> lock_weights_cache(
          weights_cache_mutex_);
      weights_cache_->delete_packed_data(executor->get_packed_data_names());
#endif
      // XNNExecutor is not trivially destructible. Since this was constructed
      // manually in init(), we must destroy it manually here.
      executor->~XNNExecutor();
    }
  }

 private:
  // This is a global workspace for all delegate instances.
  mutable std::mutex workspace_mutex_;
  std::unique_ptr<xnn_workspace, decltype(&xnn_release_workspace)> workspace_{
      nullptr,
      &xnn_release_workspace};

  // Weights cache is global to all delegate instances.
  mutable std::mutex weights_cache_mutex_;
  std::unique_ptr<XNNWeightsCache> weights_cache_ =
      std::make_unique<XNNWeightsCache>();

  // Lock Hiearchy for Mutexes:
  // workspace_mutex_
  // weights_cache_mutex_
};

namespace {
auto cls = XnnpackBackend();
Backend backend{"XnnpackBackend", &cls};
static auto success_with_compiler = register_backend(backend);
} // namespace

} // namespace backends
} // namespace executorch
