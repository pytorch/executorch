/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCommon.h>
namespace torch {
namespace executor {
namespace qnn {
QnnBackend::~QnnBackend() {
  const QnnInterface& qnn_interface = implementation_.GetQnnInterface();
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  if (nullptr != handle_) {
    QNN_EXECUTORCH_LOG_INFO("Destroy Qnn backend");
    error = qnn_interface.qnn_backend_free(handle_);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to free QNN "
          "backend_handle. Backend "
          "ID %u, error %d",
          qnn_interface.GetBackendId(),
          QNN_GET_ERROR_CODE(error));
    }
    handle_ = nullptr;
  }
}

Error QnnBackend::Configure() {
  // create qnn backend
  const QnnInterface& qnn_interface = implementation_.GetQnnInterface();
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  std::vector<const QnnBackend_Config_t*> temp_backend_config;
  ET_CHECK_OR_RETURN_ERROR(
      MakeConfig(temp_backend_config) == Error::Ok,
      Internal,
      "Fail to make backend config.");

  error = qnn_interface.qnn_backend_create(
      logger_->GetHandle(),
      temp_backend_config.empty() ? nullptr : temp_backend_config.data(),
      &handle_);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Failed to create "
        "backend_handle for Backend "
        "ID %u, error=%d",
        qnn_interface.GetBackendId(),
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }
  return Error::Ok;
}
} // namespace qnn
} // namespace executor
} // namespace torch
