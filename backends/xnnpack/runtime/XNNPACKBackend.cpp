/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/runtime/FlatbufferGraphBuilder.h>
#include <executorch/backends/xnnpack/runtime/XNNCompiler.h>
#include <executorch/backends/xnnpack/runtime/XNNPACKBackend.h>
#include <executorch/backends/xnnpack/runtime/XNNWeightsCache.h>
#include <executorch/backends/xnnpack/runtime/XNNWorkspace.h>
#include <executorch/backends/xnnpack/runtime/XnnpackBackendOptions.h>
#include <executorch/backends/xnnpack/runtime/executor/executor.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/executor/pte_data_map.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

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

struct XnnpackDelegateHandle {
  bool is_graph_runtime;
  // Legacy path: XNNExecutor placed via runtime allocator.
  xnnpack::delegate::XNNExecutor* legacy_executor = nullptr;
  // Graph path: heap-allocated Executor.
  xnnpack::executor::Executor* graph_executor = nullptr;
  std::vector<uint32_t> input_external_ids;
  std::vector<uint32_t> output_external_ids;
};

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
  }

  bool is_available() const override {
    return xnn_status_success == xnn_initialize(/*allocator=*/nullptr);
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
    auto* handle = context.get_runtime_allocator()
                       ->allocateInstance<XnnpackDelegateHandle>();
    if (handle == nullptr) {
      return Error::MemoryAllocationFailed;
    }
    new (handle) XnnpackDelegateHandle();

    bool use_graph_runtime = options_.resolve_graph_runtime(context);
    handle->is_graph_runtime = use_graph_runtime;

    if (use_graph_runtime) {
      auto result = xnnpack::FlatbufferGraphBuilder::build(
          processed->data(), processed->size());
      processed->Free();

      auto* executor = new xnnpack::executor::Executor(
          xnnpack::executor::Executor::build(result.graph));
      handle->graph_executor = executor;
      handle->input_external_ids = std::move(result.input_external_ids);
      handle->output_external_ids = std::move(result.output_external_ids);
      return handle;
    }

    auto executor = context.get_runtime_allocator()
                        ->allocateInstance<xnnpack::delegate::XNNExecutor>();
    if (executor == nullptr) {
      handle->~XnnpackDelegateHandle();
      return Error::MemoryAllocationFailed;
    }

    const NamedDataMap* named_data_map = context.get_named_data_map();

    auto program_id =
        reinterpret_cast<uintptr_t>(context.get_runtime_allocator());
    auto sharing_mode_result = options_.resolve_sharing_mode(context);
    if (!sharing_mode_result.ok()) {
      handle->~XnnpackDelegateHandle();
      return sharing_mode_result.error();
    }
    auto workspace_result =
        options_.workspace_manager().get_or_create_workspace(
            program_id, sharing_mode_result.get());
    if (!workspace_result.ok()) {
      handle->~XnnpackDelegateHandle();
      return workspace_result.error();
    }
    auto workspace = workspace_result.get();

    bool use_weight_cache = options_.resolve_weight_cache(context);
    if (use_weight_cache) {
      const std::lock_guard<std::mutex> lock_weight_cache(weights_cache_mutex_);
      weights_cache_->initialize_for_runtime(
          context.get_runtime_allocator(), named_data_map);
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
        weights_cache_.get(),
        workspace_ptr,
        named_data_map);
    // This backend does not need its processed data after compiling the model.
    processed->Free();

    if (err != Error::Ok) {
      executor->~XNNExecutor();
      handle->~XnnpackDelegateHandle();
      ET_LOG(
          Error, "XNNCompiler::compileModel failed: 0x%x", (unsigned int)err);
      return err;
    }
    handle->legacy_executor = executor;
    return handle;
  }

  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* handle,
      Span<EValue*> args) const override {
    auto* delegate = static_cast<XnnpackDelegateHandle*>(handle);

    if (delegate->is_graph_runtime) {
      return execute_graph(delegate, args);
    }

    auto executor = delegate->legacy_executor;

    std::unique_lock<std::mutex> lock_weights_cache(
        weights_cache_mutex_, std::defer_lock);
    if (executor->uses_weight_cache()) {
      lock_weights_cache.lock();
    }

    auto [raii_lock, _] = executor->get_workspace()->acquire();

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
      auto* delegate = static_cast<XnnpackDelegateHandle*>(handle);

      if (delegate->is_graph_runtime) {
        delete delegate->graph_executor;
        delegate->~XnnpackDelegateHandle();
        return;
      }

      auto executor = delegate->legacy_executor;

#ifdef ENABLE_XNNPACK_PROFILING
      executor->print_avg_op_timings();
#endif

      if (executor->uses_weight_cache()) {
        const std::lock_guard<std::mutex> lock_weights_cache(
            weights_cache_mutex_);
        weights_cache_->delete_packed_data(executor->get_packed_data_names());
      }

      // This is needed to serialize access to xnn_delete_runtime which is not
      // thread safe. This can heppen when multiple threads call destroy() on
      // the same backend instance. Make sure to hold onto the workspace
      // shared_ptr, as the pointer in the executor is freed, which includes
      // the mutex referenced by raii_lock.
      auto workspace = executor->get_workspace();
      auto [raii_lock, _] = workspace->acquire();

      // XNNExecutor is not trivially destructible. Since this was constructed
      // manually in init(), we must destroy it manually here.
      executor->~XNNExecutor();
      delegate->~XnnpackDelegateHandle();
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
    for (const auto& option : backend_options) {
      Error err = options_.set_option(option);
      if (err != Error::Ok) {
        return err;
      }
    }
    return Error::Ok;
  }

 private:
  Error execute_graph(
      XnnpackDelegateHandle* delegate,
      Span<EValue*> args) const {
    auto* executor = delegate->graph_executor;

    // Build input tensors from EValue args.
    std::vector<xnnpack::core::Tensor> inputs;
    inputs.reserve(delegate->input_external_ids.size());
    for (uint32_t ext_id : delegate->input_external_ids) {
      auto& et_tensor = args[ext_id]->toTensor();
      xnnpack::core::Tensor t;
      t.dtype = xnnpack::core::DType::Float32;  // TODO: map from et_tensor
      t.sizes.resize(et_tensor.dim());
      for (ssize_t d = 0; d < et_tensor.dim(); d++) {
        t.sizes[d] = static_cast<uint64_t>(et_tensor.size(d));
      }
      t.storage.data = et_tensor.mutable_data_ptr();
      t.storage.size_in_bytes = et_tensor.nbytes();
      t.storage.owner = xnnpack::core::StorageOwner::External;
      inputs.push_back(std::move(t));
    }

    auto outputs = executor->run(
        xnnpack::core::Span<xnnpack::core::Tensor>(
            inputs.data(), inputs.size()));

    // Copy output data back to EValue tensors.
    for (size_t i = 0; i < delegate->output_external_ids.size(); i++) {
      uint32_t ext_id = delegate->output_external_ids[i];
      auto& et_tensor = args[ext_id]->toTensor();
      auto& out_tensor = outputs[i];

      // Resize the output EValue tensor to match the computed shape.
      std::vector<exec_aten::SizesType> new_sizes_vec(out_tensor.sizes.size());
      for (size_t d = 0; d < out_tensor.sizes.size(); d++) {
        new_sizes_vec[d] = static_cast<exec_aten::SizesType>(out_tensor.sizes[d]);
      }
      exec_aten::ArrayRef<exec_aten::SizesType> new_sizes(
          new_sizes_vec.data(), new_sizes_vec.size());
      Error err = executorch::runtime::resize_tensor(et_tensor, new_sizes);
      if (err != Error::Ok) {
        return err;
      }

      // Copy output data if the output lives in arena memory.
      if (out_tensor.storage.data != et_tensor.mutable_data_ptr()) {
        std::memcpy(
            et_tensor.mutable_data_ptr(),
            out_tensor.storage.data,
            out_tensor.storage.size_in_bytes);
      }
    }

    return Error::Ok;
  }

  mutable xnnpack::XnnpackBackendOptions options_;

  // Weights cache is global to all delegate instances.
  mutable std::mutex weights_cache_mutex_;
  std::unique_ptr<XNNWeightsCache> weights_cache_ =
      std::make_unique<XNNWeightsCache>();

  // Lock Hiearchy for Mutexes:
  // weights_cache_mutex_
  // workspace_meta_mutex_
  // workspace_mutex_ (owned by executor)
};

namespace {
auto backend_instance = XnnpackBackend();
Backend backend{xnnpack::xnnpack_backend_key, &backend_instance};
static auto success_with_compiler = register_backend(backend);
} // namespace

} // namespace backends
} // namespace executorch
