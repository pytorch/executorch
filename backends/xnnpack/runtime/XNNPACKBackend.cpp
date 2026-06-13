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
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/executor/pte_data_map.h>
#include <executorch/runtime/platform/log.h>
#include <chrono>

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
  bool is_graph_runtime = false;
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
    auto* handle = context.get_runtime_allocator()
                       ->allocateInstance<XnnpackDelegateHandle>();
    if (handle == nullptr) {
      return Error::MemoryAllocationFailed;
    }
    new (handle) XnnpackDelegateHandle();

    bool use_graph_runtime = options_.resolve_graph_runtime(context);
    handle->is_graph_runtime = use_graph_runtime;

    if (use_graph_runtime) {
      auto t0 = std::chrono::steady_clock::now();
      const NamedDataMap* named_data_map = context.get_named_data_map();
      ET_UNWRAP(
          result,
          xnnpack::FlatbufferGraphBuilder::build(
              processed->data(), processed->size(), named_data_map));
      processed->Free();
      auto t1 = std::chrono::steady_clock::now();

      ET_UNWRAP(
          built_executor, xnnpack::executor::Executor::build(result.graph));
      auto* executor =
          new xnnpack::executor::Executor(std::move(built_executor));
      auto t2 = std::chrono::steady_clock::now();
      handle->graph_executor = executor;
      handle->input_external_ids = std::move(result.input_external_ids);
      handle->output_external_ids = std::move(result.output_external_ids);
      ET_LOG(
          Info,
          "Graph runtime init: deserialize=%lldms executor_build=%lldms",
          (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
              t1 - t0)
              .count(),
          (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
              t2 - t1)
              .count());
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
    // Hold the lock for the entire init-compile-finalize sequence to prevent
    // concurrent inits from resetting is_finalized_ or overwriting
    // named_data_map_ while compileModel is using the shared weights cache.
    std::unique_lock<std::mutex> lock_weights_cache(
        weights_cache_mutex_, std::defer_lock);
    if (use_weight_cache) {
      lock_weights_cache.lock();

      const auto& cache_path = options_.get_packed_cache_path();
      if (!cache_path.empty()) {
        weights_cache_->set_packed_cache_path(cache_path);
      }

      weights_cache_->initialize_for_runtime(
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
        weights_cache_.get(),
        workspace_ptr,
        named_data_map,
        use_weight_cache);
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

    auto workspace = executor->get_workspace();

    std::unique_lock<std::mutex> lock_weights_cache(
        weights_cache_mutex_, std::defer_lock);
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
      auto* delegate = static_cast<XnnpackDelegateHandle*>(handle);

      if (delegate->is_graph_runtime) {
        delete delegate->graph_executor;
        delegate->~XnnpackDelegateHandle();
        return;
      }

      auto executor = delegate->legacy_executor;
      auto workspace = executor->get_workspace();

      const std::lock_guard<std::mutex> lock_weights_cache(
          weights_cache_mutex_);

#ifdef ENABLE_XNNPACK_PROFILING
      executor->print_avg_op_timings();
#endif

      if (executor->uses_weight_cache()) {
        weights_cache_->delete_packed_data(executor->get_packed_data_names());
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
  Error execute_graph(XnnpackDelegateHandle* delegate, Span<EValue*> args)
      const {
    auto* executor = delegate->graph_executor;

    // Build input tensors from EValue args.
    std::vector<xnnpack::core::Tensor> inputs;
    inputs.reserve(delegate->input_external_ids.size());
    for (uint32_t ext_id : delegate->input_external_ids) {
      ET_CHECK_OR_RETURN_ERROR(
          ext_id < args.size(),
          InvalidProgram,
          "Input external id %u out of range (%zu args)",
          ext_id,
          args.size());
      auto& et_tensor = args[ext_id]->toTensor();
      xnnpack::core::Tensor t;
      // The external-value dtype is taken from the serialized graph spec; this
      // field is informational for the input wrapper. Defaulting to Float32
      // matches the supported (float) input set.
      t.dtype = xnnpack::core::DType::Float32;
      if (et_tensor.dim() == 0) {
        t.sizes = {1};
      } else {
        // Pass dims in physical (dim-order-permuted) layout so a channels-last
        // input matches the NHWC layout XNNPACK expects, mirroring the legacy
        // XNNExecutor path.
        size_t num_dims = et_tensor.dim();
        executorch::aten::DimOrderType
            dim_order[::executorch::runtime::kTensorDimensionLimit];
        ET_CHECK_OK_OR_RETURN_ERROR(ET_RUNTIME_NAMESPACE::get_dim_order(
            et_tensor, dim_order, num_dims));
        t.sizes.resize(num_dims);
        for (size_t d = 0; d < num_dims; d++) {
          t.sizes[d] = static_cast<uint64_t>(
              et_tensor.size(static_cast<int>(dim_order[d])));
        }
      }
      t.storage.data = et_tensor.mutable_data_ptr();
      t.storage.size_in_bytes = et_tensor.nbytes();
      t.storage.owner = xnnpack::core::StorageOwner::External;
      inputs.push_back(std::move(t));
    }

    ET_UNWRAP(outputs, executor->run({inputs.data(), inputs.size()}));

    ET_CHECK_OR_RETURN_ERROR(
        outputs.size() == delegate->output_external_ids.size(),
        Internal,
        "Executor produced %zu outputs, expected %zu",
        outputs.size(),
        delegate->output_external_ids.size());

    // Copy output data back to EValue tensors.
    for (size_t i = 0; i < delegate->output_external_ids.size(); i++) {
      uint32_t ext_id = delegate->output_external_ids[i];
      ET_CHECK_OR_RETURN_ERROR(
          ext_id < args.size(),
          InvalidProgram,
          "Output external id %u out of range (%zu args)",
          ext_id,
          args.size());
      auto& et_tensor = args[ext_id]->toTensor();
      auto& out_tensor = outputs[i];

      // Resize the output EValue tensor to match the computed shape. The
      // executor reports dims in XNNPACK physical (channels-last) order;
      // scatter them back to the tensor's logical order via its dim_order,
      // mirroring the legacy XNNExecutor::resize_outputs path.
      size_t num_dims = out_tensor.sizes.size();
      std::vector<executorch::aten::SizesType> new_sizes_vec(num_dims);
      executorch::aten::DimOrderType
          out_dim_order[::executorch::runtime::kTensorDimensionLimit];
      ET_CHECK_OK_OR_RETURN_ERROR(ET_RUNTIME_NAMESPACE::get_dim_order(
          et_tensor, out_dim_order, num_dims));
      for (size_t d = 0; d < num_dims; d++) {
        new_sizes_vec[out_dim_order[d]] =
            static_cast<executorch::aten::SizesType>(out_tensor.sizes[d]);
      }
      executorch::aten::ArrayRef<executorch::aten::SizesType> new_sizes(
          new_sizes_vec.data(), new_sizes_vec.size());
      ET_CHECK_OK_OR_RETURN_ERROR(
          executorch::runtime::resize_tensor(et_tensor, new_sizes));

      if (out_tensor.storage.data != et_tensor.mutable_data_ptr()) {
        ET_CHECK_OR_RETURN_ERROR(
            out_tensor.storage.size_in_bytes <= et_tensor.nbytes(),
            Internal,
            "Output %zu is %zu bytes, exceeds tensor capacity %zu",
            i,
            out_tensor.storage.size_in_bytes,
            et_tensor.nbytes());
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
