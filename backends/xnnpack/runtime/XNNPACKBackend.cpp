/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/runtime/XNNCompiler.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/profiler.h>

#include <memory>
#include <mutex>

#pragma clang diagnostic ignored "-Wglobal-constructors"

namespace executorch {
namespace backends {

using executorch::runtime::ArrayRef;
using executorch::runtime::Backend;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;

class XnnpackBackend final : public ::executorch::runtime::BackendInterface {
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
    auto executor = ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(
        context.get_runtime_allocator(), xnnpack::delegate::XNNExecutor);

    // Executor has been allocated but not constructed, ensure that runtime_ is
    // nullptr by constructing it in place here. NOTE: Since we use placement
    // new and since this type is not trivially destructible, we must call the
    // destructor manually in destroy().
    new (executor) xnnpack::delegate::XNNExecutor;
    Error err = xnnpack::delegate::XNNCompiler::compileModel(
        processed->data(),
        processed->size(),
        executor,
        context.get_runtime_allocator(),
        workspace_.get());
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

#ifdef ENABLE_XNNPACK_SHARED_WORKSPACE
    const std::lock_guard<std::mutex> lock(workspace_mutex_);
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
#ifdef ENABLE_XNNPACK_SHARED_WORKSPACE
      // This is needed to serialize access to xnn_delete_runtime which is not
      // thread safe. This can heppen when multiple threads call destroy() on
      // the same backend instance.
      const std::lock_guard<std::mutex> lock(workspace_mutex_);
#endif
      auto executor = static_cast<xnnpack::delegate::XNNExecutor*>(handle);
#ifdef ENABLE_XNNPACK_PROFILING
      executor->print_avg_op_timings();
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
};

namespace {
auto cls = XnnpackBackend();
Backend backend{"XnnpackBackend", &cls};
static auto success_with_compiler = register_backend(backend);
} // namespace

} // namespace backends
} // namespace executorch
