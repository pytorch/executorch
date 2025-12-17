/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/QnnBackendOptions.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnDlcManager.h>
#include <executorch/backends/qualcomm/runtime/backends/ir/IrBackend.h>

namespace executorch {
namespace backends {
namespace qnn {

QnnDlcManager::QnnDlcManager(
    const QnnExecuTorchContextBinary& qnn_context_blob,
    const QnnExecuTorchOptions* options)
    : qnn_context_blob_(qnn_context_blob), options_(options) {
  if (options_ == nullptr) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Fail to create QnnDlcManager, options is nullptr");
  }
}

Error QnnDlcManager::LoadQnnIrLibrary() {
  backend_bundle_ptr_->implementation =
      std::make_unique<QnnImplementation>(library_name_);
  Error ret = backend_bundle_ptr_->implementation->Load(nullptr);
  return ret;
}

Error QnnDlcManager::Create() {
  backend_bundle_ptr_->qnn_backend_ptr = std::make_unique<IrBackend>(
      backend_bundle_ptr_->implementation.get(),
      backend_bundle_ptr_->qnn_logger_ptr.get());

  backend_bundle_ptr_->qnn_device_ptr = std::make_unique<QnnDevice>(
      backend_bundle_ptr_->implementation.get(),
      backend_bundle_ptr_->qnn_logger_ptr.get());

  backend_params_ptr_->qnn_backend_cache_ptr_ =
      std::make_unique<QnnBackendCache>(qnn_context_blob_);

  backend_params_ptr_->qnn_context_ptr_ = std::make_unique<IrContext>(
      backend_bundle_ptr_->implementation.get(),
      backend_bundle_ptr_->qnn_backend_ptr.get(),
      backend_bundle_ptr_->qnn_device_ptr.get(),
      backend_params_ptr_->qnn_backend_cache_ptr_.get(),
      nullptr);

  backend_params_ptr_->qnn_graph_ptr_ = std::make_unique<QnnGraph>(
      backend_bundle_ptr_->implementation.get(),
      backend_bundle_ptr_->qnn_backend_ptr.get(),
      backend_params_ptr_->qnn_context_ptr_.get(),
      get_option(options_->profile_level()));
  backend_params_ptr_->backend_init_state_ =
      BackendInitializeState::INITIALIZED;
  return backend_bundle_ptr_->qnn_backend_ptr->VerifyQNNSDKVersion();
}

Error QnnDlcManager::Configure(const std::vector<std::string>& graph_names) {
  ET_CHECK_OR_RETURN_ERROR(
      backend_params_ptr_ != nullptr, Internal, "Failed to load Qnn backend.");
  ET_CHECK_OR_RETURN_ERROR(
      backend_params_ptr_->qnn_backend_cache_ptr_->Configure(graph_names) ==
          Error::Ok,
      Internal,
      "Fail to configure Qnn backend cache");
  ET_CHECK_OR_RETURN_ERROR(
      backend_bundle_ptr_->qnn_backend_ptr->Configure(
          options_->op_package_options()) == Error::Ok,
      Internal,
      "Fail to configure Qnn backend");
  ET_CHECK_OR_RETURN_ERROR(
      backend_params_ptr_->qnn_context_ptr_->Configure() == Error::Ok,
      Internal,
      "Fail to configure Qnn context");
  for (const std::string& graph_name :
       backend_params_ptr_->qnn_context_ptr_->GetGraphNames()) {
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_->qnn_graph_ptr_->Configure(graph_name) == Error::Ok,
        Internal,
        "Fail to configure Qnn graph");
  }
  backend_params_ptr_->backend_init_state_ =
      BackendInitializeState::INITIALIZED;

  return Error::Ok;
}

Error QnnDlcManager::SetUpDlcEnvironment(
    const Qnn_Version_t& coreApiVersion,
    const std::vector<std::string>& graph_names) {
  ET_CHECK_MSG(
      (coreApiVersion.major >= 2 && coreApiVersion.minor >= 23),
      "Qnn API version %u.%u.%u is not supported for Qnn IR backend, The minimum supported version is 2.23.0 or QNN_SDK version 2.30.0",
      coreApiVersion.major,
      coreApiVersion.minor,
      coreApiVersion.patch);

  ET_CHECK_OR_RETURN_ERROR(
      LoadQnnIrLibrary() == Error::Ok,
      Internal,
      "Fail to Load Qnn IR library.");

  backend_bundle_ptr_->qnn_logger_ptr = std::make_unique<QnnLogger>(
      backend_bundle_ptr_->implementation.get(),
      LoggingCallback,
      get_option(options_->log_level()));

  ET_CHECK_OR_RETURN_ERROR(
      Create() == Error::Ok, Internal, "Failed to load Qnn IR backend.");

  ET_CHECK_OR_RETURN_ERROR(
      Configure(graph_names) == Error::Ok,
      Internal,
      "Fail to configure IR backend.");

  return Error::Ok;
}

Error QnnDlcManager::RegisterGraphsFromDLC(
    QnnImplementation* implementation,
    QnnBackend* backend,
    QnnContext* context,
    QnnBackendCache* cache) {
  return Error::Ok;
}

void QnnDlcManager::Destroy() {
  backend_params_ptr_.reset(new BackendConfigParameters());
  backend_bundle_ptr_.reset(new QnnBackendBundle());
}

} // namespace qnn
} // namespace backends
} // namespace executorch
