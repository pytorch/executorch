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
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/executor/pte_data_map.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <utility>

#pragma clang diagnostic ignored "-Wglobal-constructors"

namespace executorch {
namespace backends {

using executorch::backends::xnnpack::WorkspaceSharingMode;
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
    workspace_sharing_mode_ = WorkspaceSharingMode::Global;
#else
    workspace_sharing_mode_ = WorkspaceSharingMode::Disabled;
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

    auto program_id =
        reinterpret_cast<uintptr_t>(context.get_runtime_allocator());
    auto workspace = ET_UNWRAP(get_or_create_workspace(program_id));

#ifdef ENABLE_XNNPACK_WEIGHTS_CACHE
    const std::lock_guard<std::mutex> lock_weight_cache(weights_cache_mutex_);
    weights_cache_->initialize_for_runtime(
        context.get_runtime_allocator(), named_data_map);
#endif

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
        weights_cache_.get(),
        workspace_ptr,
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
      EValue** args) const override {
    auto executor = static_cast<xnnpack::delegate::XNNExecutor*>(handle);

#ifdef ENABLE_XNNPACK_WEIGHTS_CACHE
    const std::lock_guard<std::mutex> lock_weights_cache(weights_cache_mutex_);
#endif

    auto [lock, _] = executor->get_workspace().acquire();

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
      auto executor = static_cast<xnnpack::delegate::XNNExecutor*>(handle);

#ifdef ENABLE_XNNPACK_PROFILING
      executor->print_avg_op_timings();
#endif

#ifdef ENABLE_XNNPACK_WEIGHTS_CACHE
      const std::lock_guard<std::mutex> lock_weights_cache(
          weights_cache_mutex_);
      weights_cache_->delete_packed_data(executor->get_packed_data_names());
#endif

      // This is needed to serialize access to xnn_delete_runtime which is not
      // thread safe. This can heppen when multiple threads call destroy() on
      // the same backend instance.
      auto [lock, _] = executor->get_workspace().acquire();

      // XNNExecutor is not trivially destructible. Since this was constructed
      // manually in init(), we must destroy it manually here.
      executor->~XNNExecutor();
    }
  }

  Error get_option(
      BackendOptionContext& context,
      executorch::runtime::Span<executorch::runtime::BackendOption>&
          backend_options) override {
    // Intentionally not locking here as it is not required.

    // Verify that the expected option key is present and modify the value
    for (size_t i = 0; i < backend_options.size(); ++i) {
      if (strcmp(
              backend_options[i].key,
              xnnpack::workspace_sharing_mode_option_key) == 0) {
        // Set the value to what was stored by set_option
        backend_options[i].value =
            static_cast<int>(workspace_sharing_mode_.load());
      }
    }

    return Error::Ok;
  }

  Error set_option(
      BackendOptionContext& context,
      const executorch::runtime::Span<executorch::runtime::BackendOption>&
          backend_options) override {
    if (backend_options.size() > 0) {
      for (const auto& option : backend_options) {
        if (strcmp(option.key, xnnpack::workspace_sharing_mode_option_key) ==
            0) {
          if (auto* val = std::get_if<int>(&option.value)) {
            if (*val < 0 ||
                *val > static_cast<int>(WorkspaceSharingMode::Count)) {
              ET_LOG(
                  Error,
                  "XNNPACK workspace sharing mode must be between 0 and %d, inclusive, but was %d.",
                  static_cast<int>(WorkspaceSharingMode::Count),
                  *val);
              return Error::InvalidArgument;
            }

            ET_LOG(
                Debug, "Setting XNNPACK workspace sharing mode to %d.", *val);
            workspace_sharing_mode_ = static_cast<WorkspaceSharingMode>(*val);
          } else {
            ET_LOG(Error, "XNNPACK workspace sharing mode must be an integer.");
            return Error::InvalidArgument;
          }
        }
      }
    }
    return Error::Ok;
  }

 private:
  // Backend options.
  std::atomic<WorkspaceSharingMode> workspace_sharing_mode_ =
      xnnpack::WorkspaceSharingMode::Disabled;

  // A mutex guarding global_workspace_ and model_workspaces_. Note that this
  // mutex only guards the top-level definitions, not the contents of the
  // workspace. The contents of the workspace are guarded by the workspace's own
  // mutex in the XNNWorkspace class.
  mutable std::mutex workspace_meta_mutex_;

  // A global workspace for all delegate instances, if global sharing is
  // enabled. Lazy initialized.
  mutable std::shared_ptr<XNNWorkspace> global_workspace_;

  // A map from program id to workspace for delegate instances, if per model
  // sharing is enabled. Workspaces are owned by the executor instances via
  // shared_ptr. They are tracked here via weak pointers to allow automatic
  // cleanup when the executors are destroyed while being retrievable when
  // instantiating new executors.
  mutable std::unordered_map<uintptr_t, std::weak_ptr<XNNWorkspace>>
      model_workspaces_;

  // Weights cache is global to all delegate instances.
  mutable std::mutex weights_cache_mutex_;
  std::unique_ptr<XNNWeightsCache> weights_cache_ =
      std::make_unique<XNNWeightsCache>();

  // Lock Hiearchy for Mutexes:
  // weights_cache_mutex_
  // workspace_meta_mutex_
  // workspace_mutex_ (owned by executor)

  // Retrieve a workspace for the given method ID, depending on the sharing
  // mode. A workspace will be created, if needed.
  Result<std::shared_ptr<XNNWorkspace>> get_or_create_workspace(
      uintptr_t program_id) const {
    auto mode = workspace_sharing_mode_.load();

    // Get or create the workspace according to the current sharing mode.
    if (mode == WorkspaceSharingMode::Disabled) {
      ET_LOG(Debug, "Instantiating workspace.");
      auto create_result = XNNWorkspace::create();
      if (!create_result.ok()) {
        return create_result.error();
      }

      return create_result.get();
    } else if (mode == WorkspaceSharingMode::PerModel) {
      return get_model_workspace(program_id);
    } else if (mode == WorkspaceSharingMode::Global) {
      return get_global_workspace();
    } else {
      ET_LOG(
          Error, "Invalid workspace sharing mode: %d.", static_cast<int>(mode));
      return Error::Internal;
    }
  }

  // Retrieve the global workspace, lazy initializing it if needed.
  Result<std::shared_ptr<XNNWorkspace>> get_global_workspace() const {
    std::scoped_lock<std::mutex> lock(workspace_meta_mutex_);

    // Lazy init global workspace.
    if (!global_workspace_) {
      ET_LOG(Debug, "Instantiating global workspace.");
      auto create_result = XNNWorkspace::create();
      if (!create_result.ok()) {
        return create_result.error();
      }
      global_workspace_ = create_result.get();
    }

    return global_workspace_;
  }

  // Get or create a workspace for the given program ID.
  Result<std::shared_ptr<XNNWorkspace>> get_model_workspace(
      uintptr_t program_id) const {
    std::scoped_lock<std::mutex> lock(workspace_meta_mutex_);

    // Check for an existing (live) workspace for this program.
    auto match = model_workspaces_.find(program_id);
    std::shared_ptr<XNNWorkspace> workspace = {};
    if (match != model_workspaces_.end()) {
      if (auto live_workspace = match->second.lock()) {
        workspace = live_workspace;
      }
    }

    // Allocate a new workspace if needed.
    if (!workspace) {
      ET_LOG(Debug, "Creating workspace for program %" PRIuPTR ".", program_id);
      auto create_result = XNNWorkspace::create();
      if (!create_result.ok()) {
        return create_result.error();
      }
      workspace = create_result.get();
      model_workspaces_.insert(
          {program_id, std::weak_ptr<XNNWorkspace>(workspace)});
    }

    return workspace;
  }
};

namespace {
auto backend_instance = XnnpackBackend();
Backend backend{xnnpack::xnnpack_backend_key, &backend_instance};
static auto success_with_compiler = register_backend(backend);
} // namespace

} // namespace backends
} // namespace executorch
