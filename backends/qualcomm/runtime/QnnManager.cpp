/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/QnnManager.h>
#include <executorch/backends/qualcomm/runtime/SharedBuffer.h>
#include <executorch/backends/qualcomm/runtime/Utils.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnImplementation.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
namespace torch {
namespace executor {
namespace qnn {
QnnManager::~QnnManager() {
  backend_params_ptr_.reset(new BackendConfigParameters());
  logger_.reset();
  qnn_loaded_backend_.TerminateAllBackends();
}

QnnManager::QnnManager(
    const QnnExecuTorchOptions* options,
    const QnnExecuTorchContextBinary& qnn_executorch_context_binary)
    : qnn_context_blob_(qnn_executorch_context_binary),
      qnn_loaded_backend_(""),
      // options' life cycle is decided by compiler specs which is
      // kept by executorch runtime framework
      // please pay attention to any potential seg fault
      options_(options) {
  QnnExecuTorchBackendType backend_type =
      options->backend_options()->backend_type();
  std::string library_path = options->library_path()->str();

  if (options->log_level() >= QnnExecuTorchLogLevel::kLogLevelInfo) {
    QNN_EXECUTORCH_LOG_INFO(
        "soc_model in soc_info: %s",
        EnumNameQcomChipset(options_->soc_info()->soc_model()));
    QNN_EXECUTORCH_LOG_INFO(
        "backend_type: %s", EnumNameQnnExecuTorchBackendType(backend_type));
    QNN_EXECUTORCH_LOG_INFO("graph_name: %s", options_->graph_name()->c_str());
    QNN_EXECUTORCH_LOG_INFO("library_path: %s", library_path.c_str());
    QNN_EXECUTORCH_LOG_INFO(
        "tensor_dump_output_path: %s",
        options_->tensor_dump_output_path()->c_str());
    QNN_EXECUTORCH_LOG_INFO(
        "log_level: %s", EnumNameQnnExecuTorchLogLevel(options_->log_level()));
    QNN_EXECUTORCH_LOG_INFO(
        "profile_level: %s",
        EnumNameQnnExecuTorchProfileLevel(options_->profile_level()));
    QNN_EXECUTORCH_LOG_INFO(
        "the size of qnn context binary: %d",
        qnn_executorch_context_binary.nbytes);
    QNN_EXECUTORCH_LOG_INFO(
        "Is on-device graph construction: %d", options->online_prepare());
    QNN_EXECUTORCH_LOG_INFO(
        "Enable shared buffer: %d", options->shared_buffer());
  }

  if (library_path.empty()) {
    switch (backend_type) {
      case QnnExecuTorchBackendType::kHtpBackend:
        library_path = htp_library_name_;
        break;
      case QnnExecuTorchBackendType::kDspBackend:
        library_path = dsp_library_name_;
        break;
      case QnnExecuTorchBackendType::kGpuBackend:
        library_path = gpu_library_name_;
        break;
      default:
        QNN_EXECUTORCH_LOG_ERROR("Unknown backend type: %d", backend_type);
        break;
    }
  }
  qnn_loaded_backend_ = QnnImplementation(library_path);
  backend_params_ptr_ = std::make_unique<BackendConfigParameters>();
}

Error QnnManager::LoadQnnLibrary() {
  Error ret = qnn_loaded_backend_.Load(nullptr);
  return ret;
}

Error QnnManager::RegisterMem(
    void* data_ptr,
    const std::shared_ptr<TensorWrapper>& tensor_wrapper) {
  SharedBuffer& shared_buffer_manager = SharedBuffer::GetSharedBufferManager();
  // Not enable shared buffer
  if (!options_->shared_buffer())
    return Error::Internal;

  if (backend_params_ptr_->qnn_mem_manager_ptr_ == nullptr) {
    QNN_EXECUTORCH_LOG_WARN(
        "Backend %s doesn't supported shared buffer.",
        EnumNameQnnExecuTorchBackendType(
            options_->backend_options()->backend_type()));
    return Error::Internal;
  }

  if (!shared_buffer_manager.IsAllocated(data_ptr)) {
    // It means two scenarios here:
    // 1. the input and output partitioned graph
    // 2. Actually, user doesn't allocate shared buffer with
    // QnnExecuTorchAllocCustomMem API
    return Error::Internal;
  } else if (backend_params_ptr_->qnn_mem_manager_ptr_->IsRegistered(
                 tensor_wrapper->GetMemHandle())) {
    if (options_->log_level() >= QnnExecuTorchLogLevel::kLogLevelInfo)
      QNN_EXECUTORCH_LOG_INFO(
          "Tensor name %s has been registered shared memory.",
          tensor_wrapper->GetName().c_str());
    return Error::Ok;
  }

  int32_t mem_fd = SharedBuffer::GetSharedBufferManager().MemToFd(data_ptr);
  if (mem_fd == -1) {
    QNN_EXECUTORCH_LOG_WARN(
        "Tensor name %s is failed to get file descriptor.",
        tensor_wrapper->GetName().c_str());
    return Error::Internal;
  }
  ET_CHECK_OR_RETURN_ERROR(
      backend_params_ptr_->qnn_mem_manager_ptr_->RegisterMem(
          tensor_wrapper, mem_fd) == Error::Ok,
      Internal,
      "Fail to register to shared memory.");

  return Error::Ok;
}

Error QnnManager::Init() {
  ET_CHECK_OR_RETURN_ERROR(
      LoadQnnLibrary() == Error::Ok, Internal, "Fail to load Qnn library");
  logger_ = std::make_unique<QnnLogger>(
      qnn_loaded_backend_, LoggingCallback, options_->log_level());
  if (backend_params_ptr_->backend_init_state_ ==
      BackendInitializeState::UNINITIALIZED) {
    QNN_EXECUTORCH_LOG_INFO(
        "Initialize Qnn backend "
        "parameters for Qnn executorch backend type %d",
        options_->backend_options()->backend_type());
    backend_params_ptr_ = QnnBackendFactory().Create(
        qnn_loaded_backend_, logger_.get(), qnn_context_blob_, options_);
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_->qnn_backend_ptr_->Configure() == Error::Ok,
        Internal,
        "Fail to configure Qnn backend");
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_->qnn_device_ptr_->Configure() == Error::Ok,
        Internal,
        "Fail to configure Qnn device");
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_->qnn_context_ptr_->Configure() == Error::Ok,
        Internal,
        "Fail to configure Qnn context");
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_->qnn_graph_ptr_->Configure() == Error::Ok,
        Internal,
        "Fail to configure Qnn graph");
    backend_params_ptr_->backend_init_state_ =
        BackendInitializeState::INITIALIZED;
  }

  return Error::Ok;
}

Error QnnManager::AllocateTensor() {
  std::vector<Qnn_Tensor_t> input_tensors =
      backend_params_ptr_->qnn_context_ptr_->GetGraphInputs();
  std::vector<Qnn_Tensor_t> output_tensors =
      backend_params_ptr_->qnn_context_ptr_->GetGraphOutputs();

  for (auto& tensor : input_tensors) {
    std::shared_ptr<TensorWrapper> tensor_wrapper = CreateTensorWrapper(tensor);
    tensor_wrapper->UpdateQnnTensorMeta(tensor);
    input_tensors_.emplace_back(std::move(tensor_wrapper));
  }

  for (auto& tensor : output_tensors) {
    std::shared_ptr<TensorWrapper> tensor_wrapper = CreateTensorWrapper(tensor);
    tensor_wrapper->UpdateQnnTensorMeta(tensor);
    if (IsTensorDump()) {
      tensor_wrapper->AllocateDataBuffer();
    }
    output_tensors_.emplace_back(std::move(tensor_wrapper));
  }
  return Error::Ok;
}

Error QnnManager::AllocateTensor(
    std::vector<std::shared_ptr<TensorWrapper>>& inputs,
    std::vector<std::shared_ptr<TensorWrapper>>& outputs) {
  input_tensors_ = std::move(inputs);
  for (auto& output_tensor : outputs) {
    if (IsTensorDump()) {
      output_tensor->AllocateDataBuffer();
    }
  }
  output_tensors_ = std::move(outputs);
  return Error::Ok;
}

Error QnnManager::Execute(
    const std::vector<Qnn_Tensor_t>& input_tensor_structs,
    std::vector<Qnn_Tensor_t>& output_tensor_structs) {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  error = backend_params_ptr_->qnn_graph_ptr_->GraphExecute(
      input_tensor_structs, output_tensor_structs);

  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "qnn_graph_execute failed. Error %d", QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }

  if (IsTensorDump()) {
    // TODO: Need to handle the graph which is partitioned.
    // Maybe we could use graph name.
    std::string dir = options_->tensor_dump_output_path()->str() + "/Result/";
    CreateDirectory(dir);
    QNN_EXECUTORCH_LOG_INFO("Dump tensor to the path: %s", dir.c_str());
    for (std::size_t out_idx = 0; out_idx < output_tensor_structs.size();
         ++out_idx) {
      const Qnn_Tensor_t& output_tensor = output_tensor_structs[out_idx];

      std::string output_path =
          dir + QNN_VER_PTR(output_tensor)->name + "_tensor.raw";

      std::ofstream fout(output_path, std::ios::binary);
      if (fout.fail()) {
        QNN_EXECUTORCH_LOG_ERROR(
            "Dump tensor name: %s Failed.", QNN_VER_PTR(output_tensor)->name);
        return Error::Internal;
      }

      fout.write(
          static_cast<const char*>(QNN_VER_PTR(output_tensor)->clientBuf.data),
          QNN_VER_PTR(output_tensor)->clientBuf.dataSize);
    }
  }

  return Error::Ok;
}

Error QnnManager::ProfileExecuteData(EventTracer* event_tracer) {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  if (options_->profile_level() != QnnExecuTorchProfileLevel::kProfileOff) {
    error =
        backend_params_ptr_->qnn_graph_ptr_->ProfileExecuteData(event_tracer);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          " Failed to profile. Error %d", QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }
  }
  return Error::Ok;
}

void QnnManager::Destroy() {
  QNN_EXECUTORCH_LOG_INFO("Destroy Qnn backend parameters");
  backend_params_ptr_.reset(new BackendConfigParameters());
  logger_.reset();

  qnn_loaded_backend_.TerminateAllBackends();
}

bool QnnManager::IsNodeSupportedByBackend(
    std::vector<std::shared_ptr<OpWrapper>>& op_wrappers) {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  for (std::shared_ptr<OpWrapper>& op_wrapper : op_wrappers) {
    for (const auto& param : op_wrapper->GetParams()) {
      // unused?
      // auto* p_tensor_param = dynamic_cast<TensorParamWrapper*>(param.get());
      if (param->PopulateQnnParam() != Error::Ok) {
        QNN_EXECUTORCH_LOG_WARN(
            "Qnn Backend op validation failed "
            "with PopulateQnnParam: %d",
            QNN_GET_ERROR_CODE(error));
        return false;
      }
    }

    error = backend_params_ptr_->qnn_backend_ptr_->BackendValidateOpConfig(
        op_wrapper->GetOpConfig());
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_WARN(
          "Qnn Backend op validation failed with error: %d",
          QNN_GET_ERROR_CODE(error));

      return false;
    }
  }
  return true;
}

Error QnnManager::Compile(
    std::vector<std::shared_ptr<OpWrapper>>& op_wrappers,
    QnnExecuTorchContextBinary& qnn_executorch_context_binary) {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  for (std::shared_ptr<OpWrapper>& op_wrapper : op_wrappers) {
    for (const auto& tensor_wrapper : op_wrapper->GetInputTensors()) {
      ET_CHECK_OR_RETURN_ERROR(
          backend_params_ptr_->qnn_graph_ptr_->EnsureTensorInQnnGraph(
              tensor_wrapper) == Error::Ok,
          Internal,
          "Tensor name %s isn't added to Qnn Graph",
          tensor_wrapper->GetName().c_str());
    }

    for (const auto& tensor_wrapper : op_wrapper->GetOutputTensors()) {
      ET_CHECK_OR_RETURN_ERROR(
          backend_params_ptr_->qnn_graph_ptr_->EnsureTensorInQnnGraph(
              tensor_wrapper) == Error::Ok,
          Internal,
          "Tensor name %s isn't added to Qnn Graph",
          tensor_wrapper->GetName().c_str());
    }

    for (const auto& param : op_wrapper->GetParams()) {
      auto* p_tensor_param = dynamic_cast<TensorParamWrapper*>(param.get());
      if (p_tensor_param != nullptr) {
        ET_CHECK_OR_RETURN_ERROR(
            backend_params_ptr_->qnn_graph_ptr_->EnsureTensorInQnnGraph(
                p_tensor_param->GetTensorWrapper()) == Error::Ok,
            Internal,
            "Param tensor name %s isn't added to Qnn Graph",
            p_tensor_param->GetName().c_str());
      }
      ET_CHECK_OR_RETURN_ERROR(
          param->PopulateQnnParam() == Error::Ok,
          Internal,
          "Fail to configure Qnn backend");
    }

    error = backend_params_ptr_->qnn_graph_ptr_->GraphAddNode(
        op_wrapper->GetOpConfig());
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to add node to Qnn Graph with error: %d",
          QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }
  }

  error = backend_params_ptr_->qnn_graph_ptr_->GraphFinalize();
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Failed to finalize Qnn Graph with error: %d",
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }

  // no need to generate extra context binary in online prepare scenario
  if (!IsOnlinePrepare()) {
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_->qnn_context_ptr_->GetContextBinary(
            qnn_executorch_context_binary) == Error::Ok,
        Internal,
        "Fail to get context binary.");
  }

  return Error::Ok;
};
} // namespace qnn
} // namespace executor
} // namespace torch

#ifdef _WIN32
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdll-attribute-on-redeclaration"
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT void* QnnExecuTorchAllocCustomMem(size_t bytes, size_t alignment) {
  using torch::executor::qnn::SharedBuffer;
  void* buffer_ptr =
      SharedBuffer::GetSharedBufferManager().AllocMem(bytes, alignment);
  return buffer_ptr;
}

EXPORT void QnnExecuTorchFreeCustomMem(void* buffer_ptr) {
  using torch::executor::qnn::SharedBuffer;
  SharedBuffer::GetSharedBufferManager().FreeMem(buffer_ptr);
}

#ifdef _WIN32
#pragma clang diagnostic pop
#endif
