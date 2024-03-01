/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/backends/QnnDeviceCommon.h>
namespace torch {
namespace executor {
namespace qnn {
QnnDevice::~QnnDevice() {
  const QnnInterface& qnn_interface = implementation_.GetQnnInterface();
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  if (nullptr != handle_) {
    QNN_EXECUTORCH_LOG_INFO("Destroy Qnn device");
    error = qnn_interface.qnn_device_free(handle_);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to free QNN "
          "device_handle. Backend "
          "ID %u, error %d",
          qnn_interface.GetBackendId(),
          QNN_GET_ERROR_CODE(error));
    }
    handle_ = nullptr;
  }
}

Error QnnDevice::Configure() {
  // create qnn device
  const QnnInterface& qnn_interface = implementation_.GetQnnInterface();
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  std::vector<const QnnDevice_Config_t*> temp_device_config;
  ET_CHECK_OR_RETURN_ERROR(
      MakeConfig(temp_device_config) == Error::Ok,
      Internal,
      "Fail to make device config.");

  error = qnn_interface.qnn_device_create(
      logger_->GetHandle(),
      temp_device_config.empty() ? nullptr : temp_device_config.data(),
      &handle_);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Failed to create "
        "device_handle for Backend "
        "ID %u, error=%d",
        qnn_interface.GetBackendId(),
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }

  ET_CHECK_OR_RETURN_ERROR(
      AfterCreateDevice() == Error::Ok,
      Internal,
      "Fail to configure performance config.");

  return Error::Ok;
}
} // namespace qnn
} // namespace executor
} // namespace torch
