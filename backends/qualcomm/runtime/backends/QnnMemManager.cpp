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

bool QnnMemManager::IsRegistered(Qnn_MemHandle_t handle, void* mem_ptr) {
  auto it = registered_map_.find(handle);
  if (it != registered_map_.end()) {
    return it->second == mem_ptr;
  }
  return false;
}

Error QnnMemManager::RegisterIonMem(
    const std::shared_ptr<TensorWrapper>& tensor_wrapper,
    int32_t mem_fd,
    void* mem_ptr) {
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
  registered_map_.insert({handle, mem_ptr});
  QNN_EXECUTORCH_LOG_INFO(
      "Tensor %s is successfully registered to ION shared memory.",
      tensor_wrapper->GetName().c_str());
  return Error::Ok;
}

Error QnnMemManager::RegisterCustomMem(
    const std::shared_ptr<TensorWrapper>& tensor_wrapper,
    int32_t mem_fd,
    void* mem_ptr,
    void* unaligned_custom_mem_base,
    size_t total_custom_mem_size,
    size_t tensor_offset) {
  const QnnInterface& qnn_interface = implementation_.GetQnnInterface();
  Qnn_MemDescriptor_t descriptor = {
      {tensor_wrapper->GetRank(), tensor_wrapper->GetDims(), nullptr},
      tensor_wrapper->GetDataType(),
      QNN_MEM_TYPE_CUSTOM,
      {{mem_fd}}};
  Qnn_MemHandle_t handle = nullptr;
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  QnnMemHtp_Descriptor_t htp_descriptor;
  htp_descriptor.type = QNN_HTP_MEM_SHARED_BUFFER;
  htp_descriptor.size = total_custom_mem_size;

  QnnHtpMem_SharedBufferConfig_t htpSharedBuffConfig = {mem_fd, tensor_offset};
  htp_descriptor.sharedBufferConfig = htpSharedBuffConfig;

  descriptor.customInfo = &htp_descriptor;

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
  registered_map_.insert({handle, mem_ptr});
  QNN_EXECUTORCH_LOG_INFO(
      "Tensor %s is successfully registered to custom shared memory.",
      tensor_wrapper->GetName().c_str());
  return Error::Ok;
}

Error QnnMemManager::PreRegisterCustomMemHandle(
    int32_t mem_fd,
    void* unaligned_custom_mem_base,
    size_t total_custom_mem_size,
    size_t tensor_offset,
    const CustomMemTensorInfo& info) {
  const QnnInterface& qnn_interface = implementation_.GetQnnInterface();
  Qnn_MemDescriptor_t descriptor = {
      {info.rank, info.shape, nullptr},
      scalar_type_to_qnn_dtype_[info.dtype],
      QNN_MEM_TYPE_CUSTOM,
      {{mem_fd}}};
  Qnn_MemHandle_t handle = nullptr;
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  QnnMemHtp_Descriptor_t htp_descriptor;
  htp_descriptor.type = QNN_HTP_MEM_SHARED_BUFFER;
  htp_descriptor.size = total_custom_mem_size;

  QnnHtpMem_SharedBufferConfig_t htpSharedBuffConfig = {mem_fd, tensor_offset};
  htp_descriptor.sharedBufferConfig = htpSharedBuffConfig;

  descriptor.customInfo = &htp_descriptor;

  error = qnn_interface.qnn_mem_register(
      context_->GetHandle(),
      &descriptor,
      /*numDescriptors=*/1,
      &handle);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_WARN(
        "PreRegisterCustomMemHandle fail", QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }

  pre_registered_handles_.insert({info, handle});
  registered_map_.insert({handle, nullptr});
  return Error::Ok;
}

void* QnnMemManager::GetPreRegisteredHandle(const CustomMemTensorInfo& info) {
  auto it = pre_registered_handles_.find(info);
  if (it == pre_registered_handles_.end()) {
    return nullptr;
  }
  return it->second;
}

Error QnnMemManager::SetMemHandle(
    const std::shared_ptr<TensorWrapper>& tensor_wrapper,
    void* mem_ptr,
    Qnn_MemHandle_t handle) {
  tensor_wrapper->SetMemHandle(handle);
  registered_map_.insert({handle, mem_ptr});
  return Error::Ok;
}

void QnnMemManager::DeRegisterMem() {
  const QnnInterface& qnn_interface = implementation_.GetQnnInterface();
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  for (auto& it : registered_map_) {
    error = qnn_interface.qnn_mem_de_register(&it.first, /*numHandles=*/1);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_WARN(
          "Failed to de-register shared memory. Error %d",
          QNN_GET_ERROR_CODE(error));
    }
  }
  registered_map_.clear();
}

} // namespace qnn
} // namespace executor
} // namespace torch
