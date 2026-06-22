/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/runtime/XNNCompiler.h>
#include <executorch/backends/xnnpack/runtime/XNNPACKBackend.h>
#include <executorch/backends/xnnpack/runtime/XNNWeightsCache.h>
#include <executorch/backends/xnnpack/runtime/XNNWorkspace.h>
#include <executorch/backends/xnnpack/runtime/XnnpackBackendOptions.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/executor/pte_data_map.h>

#include <memory>
#include <mutex>

#pragma clang diagnostic ignored "-Wglobal-constructors"

namespace executorch {
namespace backends {

using executorch::backends::xnnpack::XNNWorkspace;
using executorch::backends::xnnpack::delegate::XNNWeightsCache;
using executorch::ET_RUNTIME_NAMESPACE::Backend;
using executorch::ET_RUNTIME_NAMESPACE::BackendExecutionContext;
using executorch::ET_RUNTIME_NAMESPACE::BackendInitContext;
using executorch::ET_RUNTIME_NAMESPACE::BackendOptionContext;
using executorch::ET_RUNTIME_NAMESPACE::CompileSpec;
using executorch::ET_RUNTIME_NAMESPACE::DelegateHandle;
using executorch::ET_RUNTIME_NAMESPACE::NamedDataMap;
using executorch::runtime::ArrayRef;
using executorch::runtime::BackendOption;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;
using executorch::runtime::Span;

class XnnpackBackend final
    : public ::executorch::ET_RUNTIME_NAMESPACE::BackendInterface {
 public:
  ~XnnpackBackend() override = default;

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
    // thread safe. This can happen when multiple threads call init() on
    // the same backend instance.

    auto program_id =
        reinterpret_cast<uintptr_t>(context.get_runtime_allocator());
    auto sharing_mode_result = options_.resolve_sharing_mode(context);
    if (!sharing_mode_result.ok()) {
      return sharing_mode_result.error();
    }
    auto workspace_result =
        options_.workspace_manager().get_or_create_workspace(
            program_id, sharing_mode_result.get());
    if (!workspace_result.ok()) {
      return workspace_result.error();
    }
    auto workspace = workspace_result.get();

    bool use_weight_cache = options_.resolve_weight_cache(context);
    // Hold the lock for the entire init-compile-finalize sequence to prevent
    // concurrent inits from resetting is_finalized_ or overwriting
    // named_data_map_ while compileModel is using the shared weights cache.
    std::unique_lock<std::mutex> lock_weights_cache(
        options_.weights_cache_mutex(), std::defer_lock);
    if (use_weight_cache) {
      lock_weights_cache.lock();

      const auto& cache_path = options_.get_packed_cache_path();
      options_.weights_cache().set_packed_cache_path(cache_path);

      options_.weights_cache().initialize_for_runtime(
          context.get_runtime_allocator(), named_data_map);
      workspace->set_uses_weight_cache();
    }

    auto [workspace_lock, workspace_ptr] = workspace->acquire();

    // Executor has been allocated but not constructed, ensure that runtime_ is
    // nullptr by constructing it in place here. NOTE: Since we use placement
    // new and since this type is not trivially destructible, we must call the
    // destructor manually in destroy().
    new (executor) xnnpack::delegate::XNNExecutor(workspace);
    Error err = xnnpack::delegate::XNNCompiler::compileModel(
        processed->data(),
        processed->size(),
        executor,
        &options_.weights_cache(),
        workspace_ptr,
        named_data_map,
        use_weight_cache);
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

    auto workspace = executor->get_workspace();

    std::unique_lock<std::mutex> lock_weights_cache(
        options_.weights_cache_mutex(), std::defer_lock);
    if (executor->uses_weight_cache() || workspace->uses_weight_cache()) {
      lock_weights_cache.lock();
    }

    auto [raii_lock, _] = workspace->acquire();

    // Prepare Inputs/Outputs and Propagate Input Shapes
    Error err = executor->prepare_args(args);
    if (err != Error::Ok) {
      return err;
    }

    err = executor->forward(context);

    if (err != Error::Ok) {
      return err;
    }

    // Convert output data types if necessary (e.g., int32 -> int64 for Long)
    err = executor->convert_outputs(args);

    return err;
  }

  void destroy(DelegateHandle* handle) const override {
    if (handle != nullptr) {
      auto executor = static_cast<xnnpack::delegate::XNNExecutor*>(handle);
      auto workspace = executor->get_workspace();

      const std::lock_guard<std::mutex> lock_weights_cache(
          options_.weights_cache_mutex());

#ifdef ENABLE_XNNPACK_PROFILING
      executor->print_avg_op_timings();
#endif

      if (executor->uses_weight_cache()) {
        options_.weights_cache().delete_packed_data(
            executor->get_packed_data_names());
      }

      // This is needed to serialize access to xnn_delete_runtime which is not
      // thread safe. This can heppen when multiple threads call destroy() on
      // the same backend instance. Make sure to hold onto the workspace
      // shared_ptr, as the pointer in the executor is freed, which includes
      // the mutex referenced by raii_lock.
      auto [raii_lock, _] = workspace->acquire();

      // XNNExecutor is not trivially destructible. Since this was constructed
      // manually in init(), we must destroy it manually here.
      executor->~XNNExecutor();
    }
  }

  Error get_option(
      BackendOptionContext& context,
      Span<BackendOption>& backend_options) override {
    for (size_t i = 0; i < backend_options.size(); ++i) {
      Error err = options_.get_option(backend_options[i]);
      if (err != Error::Ok) {
        return err;
      }
    }
    return Error::Ok;
  }

  Error set_option(
      BackendOptionContext& context,
      const Span<BackendOption>& backend_options) override {
    // Process every option even if one fails — applying a `packed_cache_path`
    // and triggering `save_weight_cache_on_disk` in the same array must not
    // depend on declaration order. Capture the first error and report it
    // after the loop. All option-key dispatch — including the disk-save
    // side effect — lives inside XnnpackBackendOptions::set_option, which
    // owns the weights-cache instance and its mutex.
    Error first_err = Error::Ok;
    for (const auto& option : backend_options) {
      Error err = options_.set_option(option);
      if (err != Error::Ok && first_err == Error::Ok) {
        first_err = err;
      }
    }
    return first_err;
  }

 private:
  mutable xnnpack::XnnpackBackendOptions options_;

  // Lock hierarchy for mutexes:
  //   options_.weights_cache_mutex()
  //   workspace_meta_mutex_
  //   workspace_mutex_ (owned by executor)
};

namespace {
auto backend_instance = XnnpackBackend();
Backend backend{xnnpack::xnnpack_backend_key, &backend_instance};
static auto success_with_compiler = register_backend(backend);
} // namespace

} // namespace backends
} // namespace executorch
