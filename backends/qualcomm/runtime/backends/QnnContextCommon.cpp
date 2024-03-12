/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/backends/QnnContextCommon.h>
#include "HTP/QnnHtpContext.h"
namespace torch {
namespace executor {
namespace qnn {
QnnContext::~QnnContext() {
  const QnnInterface& qnn_interface = implementation_.GetQnnInterface();
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  if (handle_ != nullptr) {
    QNN_EXECUTORCH_LOG_INFO("Destroy Qnn context");
    error = qnn_interface.qnn_context_free(handle_, /*profile=*/nullptr);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to free QNN "
          "context_handle_. Backend "
          "ID %u, error %d",
          qnn_interface.GetBackendId(),
          QNN_GET_ERROR_CODE(error));
    }
    handle_ = nullptr;
  }
}

Error QnnContext::Configure(int spill_fill_size) {
  // create qnn context
  const QnnInterface& qnn_interface = implementation_.GetQnnInterface();
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  std::vector<const QnnContext_Config_t*> temp_context_config;
  ET_CHECK_OR_RETURN_ERROR(
      MakeConfig(temp_context_config) == Error::Ok,
      Internal,
      "Fail to make context config.");

  if (spill_fill_size > 0) {
    QnnHtpContext_CustomConfig_t context_config;
    QnnHtpContext_GroupRegistration_t group_config{nullptr};
    if (m_sf_handle == nullptr) {
      group_config.firstGroupHandle = nullptr;
    } else {
      group_config.firstGroupHandle = m_sf_handle;
    }
    group_config.maxSpillFillBuffer = spill_fill_size;
    context_config.groupRegistration = group_config;

    QnnContext_Config_t** context_config_ptr{nullptr};
    uint32_t num_context = 1;
    context_config_ptr[0] = (QnnContext_Config_t*) malloc(num_context * sizeof(QnnContext_Config_t*));
    context_config_ptr[0]->option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    context_config_ptr[0]->customConfig = reinterpret_cast<QnnContext_CustomConfig_t>(&context_config);
    temp_context_config.push_back(context_config_ptr[0]);
  }

  if (cache_->GetCacheState() == QnnBackendCache::DESERIALIZE) {
    const QnnExecuTorchContextBinary& qnn_context_blob =
        cache_->GetQnnContextBlob();
    error = qnn_interface.qnn_context_create_from_binary(
        backend_->GetHandle(),
        device_->GetHandle(),
        temp_context_config.empty() ? nullptr : temp_context_config.data(),
        qnn_context_blob.buffer,
        qnn_context_blob.nbytes,
        &handle_,
        /*profile=*/nullptr);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Can't create context from "
          "binary. Error %d.",
          QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }
  } else if (
      cache_->GetCacheState() == QnnBackendCache::SERIALIZE ||
      cache_->GetCacheState() == QnnBackendCache::ONLINE_PREPARE) {
    error = qnn_interface.qnn_context_create(
        backend_->GetHandle(),
        device_->GetHandle(),
        temp_context_config.empty() ? nullptr : temp_context_config.data(),
        &handle_);

    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to create QNN context for Backend "
          "ID %u, error=%d",
          qnn_interface.GetBackendId(),
          QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }
  } else {
    QNN_EXECUTORCH_LOG_ERROR("QNN context cache is invalid.");
    return Error::Internal;
  }
  if (spill_fill_size > 0 && m_sf_handle == nullptr) {
     m_sf_handle = handle_;
  }
  return Error::Ok;
}

Error QnnContext::GetContextBinary(
    QnnExecuTorchContextBinary& qnn_executorch_context_binary) {
  const QnnInterface& qnn_interface = implementation_.GetQnnInterface();
  Qnn_ContextBinarySize_t binary_size = 0;
  Qnn_ContextBinarySize_t bytes_written = 0;
  Qnn_ErrorHandle_t error =
      qnn_interface.qnn_context_get_binary_size(handle_, &binary_size);
  if (error == QNN_SUCCESS) {
    binary_buffer_.reserve(binary_size);
    error = qnn_interface.qnn_context_get_binary(
        handle_, binary_buffer_.data(), binary_size, &bytes_written);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Can't get graph binary to be saved to "
          "cache. Error %d",
          QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    } else {
      if (binary_size < bytes_written) {
        QNN_EXECUTORCH_LOG_ERROR(
            "Illegal written buffer size [%d] bytes. Cannot "
            "exceed allocated memory of [%d] bytes",
            bytes_written,
            binary_size);
        return Error::Internal;
      }
      qnn_executorch_context_binary.buffer = binary_buffer_.data();
      qnn_executorch_context_binary.nbytes = bytes_written;
    }
  } else {
    QNN_EXECUTORCH_LOG_ERROR(
        "Can't determine the size of "
        "graph binary to be saved to cache. Error %d",
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }
  return Error::Ok;
}
} // namespace qnn
} // namespace executor
} // namespace torch
