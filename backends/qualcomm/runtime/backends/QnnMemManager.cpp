/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/backends/QnnMemManager.h>

namespace torch {
namespace executor {
namespace qnn {

bool QnnMemManager::IsRegistered(Qnn_MemHandle_t handle) {
  return registered_set_.count(handle) != 0U;
}

Error QnnMemManager::RegisterMem(
    const std::shared_ptr<TensorWrapper>& tensor_wrapper,
    int32_t mem_fd) {
  const QnnInterface& qnn_interface = implementation_.GetQnnInterface();
  Qnn_MemDescriptor_t descriptor = {
      {tensor_wrapper->GetRank(), tensor_wrapper->GetDims(), nullptr},
      tensor_wrapper->GetDataType(),
      QNN_MEM_TYPE_ION,
      {{mem_fd}}};
  Qnn_MemHandle_t handle = nullptr;
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  error = qnn_interface.qnn_mem_register(
      context_->GetHandle(),
      &descriptor,
      /*numDescriptors=*/1,
      &handle);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_WARN(
        "Tensor %s is failed to register shared memory. Error %d",
        tensor_wrapper->GetName().c_str(),
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }
  tensor_wrapper->SetMemHandle(handle);
  registered_set_.insert(handle);
  QNN_EXECUTORCH_LOG_INFO(
      "Tensor %s is successfully registered to shared memory.",
      tensor_wrapper->GetName().c_str());
  return Error::Ok;
}

void QnnMemManager::DeRegisterMem() {
  const QnnInterface& qnn_interface = implementation_.GetQnnInterface();
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  for (auto& mem_handle : registered_set_) {
    error = qnn_interface.qnn_mem_de_register(&mem_handle, /*numHandles=*/1);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_WARN(
          "Failed to de-register shared memory. Error %d",
          QNN_GET_ERROR_CODE(error));
    }
  }
  registered_set_.clear();
}

} // namespace qnn
} // namespace executor
} // namespace torch
