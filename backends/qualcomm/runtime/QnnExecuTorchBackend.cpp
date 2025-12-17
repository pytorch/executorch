/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/aot/wrappers/TensorWrapper.h>
#include <executorch/backends/qualcomm/qc_compiler_spec_generated.h>
#include <executorch/backends/qualcomm/runtime/QnnBackendOptions.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorchBackend.h>
#include <executorch/backends/qualcomm/runtime/QnnManager.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnCustomProtocol.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
namespace executorch {
namespace backends {
namespace qnn {

using namespace qnn_delegate;
using executorch::runtime::ArrayRef;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Result;
using executorch::runtime::Span;

// ========== Public method implementations =========================
constexpr const char* QNN_COMPILE_SPEC = "qnn_compile_spec";
Result<DelegateHandle*> QnnExecuTorchBackend::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {
  // covert SizedBuffer to qnn ExecuTorch option
  QnnExecuTorchContextBinary qnn_context_blob;
  const qnn_delegate::QnnExecuTorchOptions* qnn_executorch_options = nullptr;
  auto [status, signature, ctx_size, ctx_bin] =
      QnnContextCustomProtocol().DeserializeContextCustomBuffer(
          const_cast<void*>(processed->data()));
  if (status == Error::Ok) {
    QNN_EXECUTORCH_LOG_INFO(
        "Deserializing processed data using QnnContextCustomProtocol");
    // After this stage, qnn_context_blob.nbytes & qnn_context_blob.buffer will
    // only store qnn_context_binary.
    qnn_context_blob.nbytes = ctx_size;
    qnn_context_blob.buffer = ctx_bin;
  } else {
    // This buffer will be verified again in QnnBackendCache.
    QNN_EXECUTORCH_LOG_INFO("Deserializing processed data using Dlc");
    qnn_context_blob.buffer = const_cast<void*>(processed->data());
    qnn_context_blob.nbytes = processed->size();
  }

  // convert CompileSpec to qnn ExecuTorch option
  for (auto& compile_spec : compile_specs) {
    if (std::strcmp(compile_spec.key, QNN_COMPILE_SPEC) == 0)
      qnn_executorch_options =
          GetQnnExecuTorchOptions(compile_spec.value.buffer);
    else
      QNN_EXECUTORCH_LOG_WARN("unknown argument: %s", compile_spec.key);
  }

  // Create QnnManager
  MemoryAllocator* runtime_allocator = context.get_runtime_allocator();
  QnnManager* qnn_manager = runtime_allocator->allocateInstance<QnnManager>();
  if (qnn_manager == nullptr) {
    return Error::MemoryAllocationFailed;
  }

  // NOTE: Since we use placement new and since this type is not trivially
  // destructible, we must call the destructor manually in destroy().
  new (qnn_manager) QnnManager(qnn_executorch_options, qnn_context_blob);
  // TODO: this is a temporal solution for multi-graph support, will be
  //       removed once framework starts to accept runtime configuration
  // ---
  // check if current context binary has already been initialized
  // return cached one for reducing memory footprint

  auto iter = delegate_map_.find(signature);
  if (iter != delegate_map_.end()) {
    QNN_EXECUTORCH_LOG_INFO(
        "Use cached delegate handle for current method: %s",
        context.get_method_name());
    return iter->second;
  }

  ET_CHECK_OR_RETURN_ERROR(
      qnn_manager->InitBackend() == Error::Ok,
      Internal,
      "Fail to initialize Qnn Manager");
  ET_CHECK_OR_RETURN_ERROR(
      qnn_manager->InitContext() == Error::Ok,
      Internal,
      "Fail to initialize Qnn Manager");

  if (qnn_manager->IsOnlinePrepare()) {
    ET_CHECK_OR_RETURN_ERROR(
        qnn_manager->CompileDlc() == Error::Ok,
        Internal,
        "Fail to compile binary in Dlc format");
  } else {
    for (const std::string& graph_name : qnn_manager->GetGraphNames()) {
      ET_CHECK_OR_RETURN_ERROR(
          qnn_manager->AllocateTensor(graph_name) == Error::Ok,
          Internal,
          "Fail to allocate tensor");
    }
  }
  add_cached_delegate(signature, qnn_manager);
  // This backend does not need its processed data after Init.
  processed->Free();
  return qnn_manager;
}

Error QnnExecuTorchBackend::execute(
    BackendExecutionContext& context,
    DelegateHandle* handle,
    Span<EValue*> args) const {
  ET_CHECK_OR_RETURN_ERROR(
      delegate_map_rev_.count(handle) != 0,
      Internal,
      "DelegateHandle has been deleted");
  QnnManager* qnn_manager = static_cast<QnnManager*>(handle);

  std::string method_name = context.get_method_name();
  std::vector<std::shared_ptr<TensorWrapper>> input_tensors =
      qnn_manager->GetGraphInputs(method_name);
  std::vector<std::shared_ptr<TensorWrapper>> output_tensors =
      qnn_manager->GetGraphOutputs(method_name);
  std::vector<Qnn_Tensor_t> input_tensor_structs;
  std::vector<Qnn_Tensor_t> output_tensor_structs;

  int args_index = 0;
  input_tensor_structs.reserve(input_tensors.size());
  for (const auto& input_tensor : input_tensors) {
    if (input_tensor->GetName().find("mutbuf_") == std::string::npos) {
      if (qnn_manager->RegisterMem(
              args[args_index]->toTensor().mutable_data_ptr(), input_tensor) !=
          Error::Ok) {
        // update data ptr only should be fine
        input_tensor->FillDataBuffer(
            args[args_index]->toTensor().const_data_ptr(),
            false /* copy_data */);
        // use the real input shape instead of nominal one to make sure
        // dynamic shape is functional
        auto dims = args[args_index]->toTensor().sizes();
        input_tensor->SetDims(dims.data(), dims.size());
      }
      args_index++;
    }
    input_tensor_structs.emplace_back(input_tensor->CloneTensorStruct());
  }

  for (const auto& output_tensor : output_tensors) {
    // pos=0 limits the search to the prefix
    if (output_tensor->GetName().rfind("output_", 0) == 0 &&
        output_tensor->GetName().find("mutbuf_") == std::string::npos) {
      void* mutable_data_ptr = args[args_index]->toTensor().mutable_data_ptr();
      if (qnn_manager->RegisterMem(mutable_data_ptr, output_tensor) !=
          Error::Ok) {
        output_tensor->FillDataBuffer(mutable_data_ptr, false /* copy_data */);
      }
      args_index++;
    }
    output_tensor_structs.push_back(output_tensor->CloneTensorStruct());
  }

  ET_CHECK_OR_RETURN_ERROR(
      qnn_manager->Execute(
          method_name,
          input_tensor_structs,
          output_tensor_structs,
          context.event_tracer()) == Error::Ok,
      Internal,
      "Fail to execute graph");
  ET_CHECK_OR_RETURN_ERROR(
      qnn_manager->ProfileExecuteData(method_name, context.event_tracer()) ==
          Error::Ok,
      Internal,
      "Fail to profile graph");

  return Error::Ok;
}

void QnnExecuTorchBackend::destroy(DelegateHandle* handle) const {
  if (handle != nullptr && delegate_map_rev_.count(handle)) {
    QnnManager* qnn_manager = static_cast<QnnManager*>(handle);
    qnn_manager->Destroy();
    erase_cached_delegate(handle);
  }
}

executorch::runtime::Error QnnExecuTorchBackend::set_option(
    executorch::runtime::BackendOptionContext& context,
    const executorch::runtime::Span<executorch::runtime::BackendOption>&
        backend_options) {
  std::lock_guard<std::mutex> guard(runtime_option_mutex_);
  size_t matches = backend_options.size();
  for (const auto& option : backend_options) {
    if (strcmp(option.key, QNN_RUNTIME_LOG_LEVEL) == 0) {
      if (auto* val = std::get_if<int>(&option.value)) {
        qnn_runtime_log_level_.value = *val;
        qnn_runtime_log_level_.is_set = true;
      }
    } else if (strcmp(option.key, QNN_RUNTIME_HTP_PERFORMANCE_MODE) == 0) {
      if (auto* val = std::get_if<int>(&option.value)) {
        qnn_runtime_performance_mode_.value = *val;
        qnn_runtime_performance_mode_.is_set = true;
      }
    } else if (strcmp(option.key, QNN_RUNTIME_PROFILE_LEVEL) == 0) {
      if (auto* val = std::get_if<int>(&option.value)) {
        qnn_runtime_profile_level_.value = *val;
        qnn_runtime_profile_level_.is_set = true;
      }
    } else {
      ET_LOG(
          Error,
          "Unable to set the following runtime option for QnnExecuTorchBackend: %s.",
          option.key);
      matches--;
    }
  }

  ET_CHECK_OR_RETURN_ERROR(
      matches == backend_options.size(),
      Internal,
      "Some set options are not supported by QnnExecuTorchBackend. %zu options provided but only %zu is supported.",
      backend_options.size(),
      matches);

  return Error::Ok;
}

executorch::runtime::Error QnnExecuTorchBackend::get_option(
    executorch::runtime::BackendOptionContext& context,
    executorch::runtime::Span<executorch::runtime::BackendOption>&
        backend_options) {
  size_t matches = backend_options.size();
  for (size_t i = 0; i < backend_options.size(); ++i) {
    // Set the value to what was stored by set_option
    if (strcmp(backend_options[i].key, QNN_RUNTIME_LOG_LEVEL) == 0 &&
        qnn_runtime_log_level_.is_set) {
      backend_options[i].value = qnn_runtime_log_level_.value;
    } else if (
        strcmp(backend_options[i].key, QNN_RUNTIME_HTP_PERFORMANCE_MODE) == 0 &&
        qnn_runtime_performance_mode_.is_set) {
      backend_options[i].value = qnn_runtime_performance_mode_.value;
    } else if (
        strcmp(backend_options[i].key, QNN_RUNTIME_PROFILE_LEVEL) == 0 &&
        qnn_runtime_profile_level_.is_set) {
      backend_options[i].value = qnn_runtime_profile_level_.value;
    } else {
      // either runtime never called set_option or key does not exist
      matches--;
    }
  }

  if (matches != backend_options.size()) {
    return Error::Internal;
  }
  return Error::Ok;
}

bool QnnExecuTorchBackend::is_available() const {
  return true;
}

void QnnExecuTorchBackend::add_cached_delegate(
    const std::int64_t& signature,
    executorch::runtime::DelegateHandle* handle) const {
  std::lock_guard<std::mutex> guard(mutex_);
  delegate_map_[signature] = handle;
  delegate_map_rev_[handle] = signature;
}

void QnnExecuTorchBackend::erase_cached_delegate(
    executorch::runtime::DelegateHandle* handle) const {
  std::lock_guard<std::mutex> guard(mutex_);
  auto iter = delegate_map_rev_.find(handle);
  if (iter == delegate_map_rev_.end()) {
    return;
  }
  delegate_map_.erase(iter->second);
  delegate_map_rev_.erase(handle);
}

namespace {
auto cls = QnnExecuTorchBackend();
executorch::runtime::Backend backend{QNN_BACKEND, &cls};
static auto success_with_compiler = register_backend(backend);
} // namespace
} // namespace qnn
} // namespace backends
} // namespace executorch
