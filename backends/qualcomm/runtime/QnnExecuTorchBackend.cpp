/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/aot/wrappers/TensorWrapper.h>
#include <executorch/backends/qualcomm/qc_compiler_spec_generated.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorchBackend.h>
#include <executorch/backends/qualcomm/runtime/QnnManager.h>

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

// ========== Public method implementations =========================
constexpr const char* QNN_COMPILE_SPEC = "qnn_compile_spec";
Result<DelegateHandle*> QnnExecuTorchBackend::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {
  // covert SizedBuffer to qnn ExecuTorch option
  QnnExecuTorchContextBinary qnn_context_blob;
  const qnn_delegate::QnnExecuTorchOptions* qnn_executorch_options = nullptr;

  qnn_context_blob.buffer = const_cast<void*>(processed->data());
  qnn_context_blob.nbytes = processed->size();

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
  QnnManager* qnn_manager =
      ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(runtime_allocator, QnnManager);

  // NOTE: Since we use placement new and since this type is not trivially
  // destructible, we must call the destructor manually in destroy().
  new (qnn_manager) QnnManager(qnn_executorch_options, qnn_context_blob);

  // TODO: this is a temporal solution for multi-graph support, will be
  //       removed once framework starts to accept runtime configuration
  // ---
  // check if current context binary has already been initialized
  // return cached one for reducing memory footprint
  std::string signature = qnn_manager->GetBinarySignature();
  auto iter = delegate_map_.find(signature);
  if (iter != delegate_map_.end()) {
    QNN_EXECUTORCH_LOG_INFO(
        "Use cached delegate handle for current method: %s",
        context.get_method_name());
    return iter->second;
  }

  ET_CHECK_OR_RETURN_ERROR(
      qnn_manager->Init() == Error::Ok,
      Internal,
      "Fail to initialize Qnn Manager");

  if (qnn_manager->IsOnlinePrepare()) {
    ET_CHECK_OR_RETURN_ERROR(
        qnn_manager->CompileQcir() == Error::Ok,
        Internal,
        "Fail to compile binary in qcir format");
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
    EValue** args) const {
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

  input_tensor_structs.reserve(input_tensors.size());
  for (int i = 0; i < input_tensors.size(); ++i) {
    if (qnn_manager->RegisterMem(
            args[i]->toTensor().mutable_data_ptr(), input_tensors[i]) !=
        Error::Ok) {
      // update data ptr only should be fine
      input_tensors[i]->FillDataBuffer(
          args[i]->toTensor().const_data_ptr(), false /* copy_data */);
    }
    input_tensor_structs.push_back(input_tensors[i]->CloneTensorStruct());
  }

  int output_index = input_tensors.size();
  for (const auto& output_tensor : output_tensors) {
    // pos=0 limits the search to the prefix
    if (output_tensor->GetName().rfind("output_", 0) == 0) {
      void* mutable_data_ptr =
          args[output_index]->toTensor().mutable_data_ptr();
      if (qnn_manager->RegisterMem(mutable_data_ptr, output_tensor) !=
          Error::Ok) {
        output_tensor->FillDataBuffer(mutable_data_ptr, false /* copy_data */);
      }
      output_index++;
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

bool QnnExecuTorchBackend::is_available() const {
  return true;
}

void QnnExecuTorchBackend::add_cached_delegate(
    const std::string& signature,
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
executorch::runtime::Backend backend{"QnnBackend", &cls};
static auto success_with_compiler = register_backend(backend);
} // namespace
} // namespace qnn
} // namespace backends
} // namespace executorch
