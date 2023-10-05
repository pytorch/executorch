/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/QnnManager.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnImplementation.h>

#include <cstdlib>
#include <cstring>
namespace torch {
namespace executor {
namespace qnn {
QnnManager::~QnnManager() {
  backend_params_ptr_.reset(new BackendConfigParameters());
  logger_.reset();
  qnn_loaded_backend_.TerminateAllBackends();
}

QnnManager::QnnManager(const QnnExecuTorchOptions* options)
    : backend_type_(options->backend_type),
      library_path_(options->library_path),
      skel_library_dir_(options->skel_library_dir),
      graph_name_(options->graph_name),
      htp_options_(options->htp_options),
      log_level_(options->log_level),
      qnn_context_blob_(options->qnn_context_blob),
      qnn_loaded_backend_(library_path_) {
  if (!skel_library_dir_.empty()) {
    setenv("ADSP_LIBRARY_PATH", skel_library_dir_.c_str(), /*overwrite=*/1);
  }
  if (library_path_.empty()) {
    switch (backend_type_) {
      case kHtpBackend:
        library_path_ = htp_library_name_;
        break;
      case kDspBackend:
        library_path_ = dsp_library_name_;
        break;
      case kGpuBackend:
        library_path_ = gpu_library_name_;
        break;
      default:
        QNN_EXECUTORCH_LOG(
            kLogLevelError,
            "[Qnn ExecuTorch] Unknown backend type: %s",
            backend_type_);
        break;
    }
  }
  qnn_loaded_backend_ = QnnImplementation(library_path_);
  backend_params_ptr_ = std::make_unique<BackendConfigParameters>();
}

Error QnnManager::LoadQnnLibrary() {
  Error ret = qnn_loaded_backend_.Load(nullptr);
  return ret;
}

Error QnnManager::Init() {
  ET_CHECK_OR_RETURN_ERROR(
      LoadQnnLibrary() == Error::Ok, Internal, "Fail to load Qnn library");
  logger_ = std::make_unique<QnnLogger>(
      qnn_loaded_backend_, LoggingCallback, log_level_);
  if (backend_params_ptr_->backend_init_state_ ==
      BackendInitializeState::UNINITIALIZED) {
    QNN_EXECUTORCH_LOG(
        kLogLevelInfo,
        "[Qnn ExecuTorch] Initialize Qnn backend "
        "parameters for Qnn executorch backend type %d",
        backend_type_);
    backend_params_ptr_ = QnnBackendFactory().Create(
        qnn_loaded_backend_,
        logger_.get(),
        qnn_context_blob_,
        backend_type_,
        graph_name_,
        htp_options_);
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
    output_tensors_.emplace_back(std::move(tensor_wrapper));
  }
  return Error::Ok;
}

Error QnnManager::Execute(
    const std::vector<Qnn_Tensor_t>& input_tensor_structs,
    std::vector<Qnn_Tensor_t>& output_tensor_structs) {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  error = backend_params_ptr_->qnn_graph_ptr_->GraphExecute(
      input_tensor_structs, output_tensor_structs);

  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG(
        kLogLevelError,
        "[Qnn ExecuTorch] qnn_graph_execute failed. Error %d",
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }

  return Error::Ok;
}

void QnnManager::Destroy() {
  QNN_EXECUTORCH_LOG(
      kLogLevelInfo, "[Qnn ExecuTorch] Destroy Qnn backend parameters");
  backend_params_ptr_.reset(new BackendConfigParameters());
  logger_.reset();

  qnn_loaded_backend_.TerminateAllBackends();
}

bool QnnManager::IsAvailable() {
  return true;
}

bool QnnManager::IsNodeSupportedByBackend(
    std::vector<std::shared_ptr<OpWrapper>>& op_wrappers) {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  for (std::shared_ptr<OpWrapper>& op_wrapper : op_wrappers) {
    for (const auto& param : op_wrapper->GetParams()) {
      // unused?
      // auto* p_tensor_param = dynamic_cast<TensorParamWrapper*>(param.get());
      if (param->PopulateQnnParam() != Error::Ok) {
        QNN_EXECUTORCH_LOG(
            kLogLevelWarn,
            "[Qnn ExecuTorch] Qnn Backend op validation failed "
            "with PopulateQnnParam: %d",
            QNN_GET_ERROR_CODE(error));
        return false;
      }
    }

    error = backend_params_ptr_->qnn_backend_ptr_->BackendValidateOpConfig(
        op_wrapper->GetOpConfig());
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG(
          kLogLevelWarn,
          "[Qnn ExecuTorch] Qnn Backend op validation failed with error: %d",
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
      QNN_EXECUTORCH_LOG(
          kLogLevelError,
          "[Qnn ExecuTorch] Failed to add node to Qnn Graph with error: %d",
          QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }
  }

  error = backend_params_ptr_->qnn_graph_ptr_->GraphFinalize();
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG(
        kLogLevelError,
        "[Qnn ExecuTorch] Failed to finalize Qnn Graph with error: %d",
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }

  ET_CHECK_OR_RETURN_ERROR(
      backend_params_ptr_->qnn_context_ptr_->GetContextBinary(
          qnn_executorch_context_binary) == Error::Ok,
      Internal,
      "Fail to get context binary.");
  return Error::Ok;
};
} // namespace qnn
} // namespace executor
} // namespace torch

QnnExecuTorchOptions QnnExecuTorchOptionsDefault() {
  QnnExecuTorchOptions options;
  std::memset(
      &options,
      0,
      sizeof(QnnExecuTorchOptions)); // clean struct padding
  options = QNN_EXECUTORCH_OPTION_INIT;
  return options;
}
