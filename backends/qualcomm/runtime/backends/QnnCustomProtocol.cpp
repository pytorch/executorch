/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/aot/ir/qcir_utils.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnCustomProtocol.h>

namespace executorch {
namespace backends {
namespace qnn {

void QnnQcirCustomProtocol::BuildQcirCustomBuffer(
    const QnnExecuTorchContextBinary& qcir_binary,
    const std::vector<uint8_t>& tensor_data) {
  if (qnn_custom_buffer_.size() == 0) {
    uint8_t magic_number_proto_size = sizeof(magic_number_);
    uint8_t qcir_fbs_proto_size = sizeof(qcir_fbs_size_);
    uint8_t tensor_proto_size = sizeof(tensor_size_);

    uint64_t buffer_size = magic_number_proto_size + qcir_fbs_proto_size +
        tensor_proto_size + qcir_fbs_size_ + tensor_size_;
    qnn_custom_buffer_.resize(buffer_size, 0);

    size_t pos = 0;
    // magic number itself
    std::memcpy(
        qnn_custom_buffer_.data(), &magic_number_, magic_number_proto_size);
    pos += magic_number_proto_size;

    // size of qcir_fbs, should be 4 bytes
    std::memcpy(
        qnn_custom_buffer_.data() + pos, &qcir_fbs_size_, qcir_fbs_proto_size);
    pos += qcir_fbs_proto_size;

    // size of tensor, should be 8 bytes
    std::memcpy(
        qnn_custom_buffer_.data() + pos, &tensor_size_, tensor_proto_size);
    pos += tensor_proto_size;

    // qcir.fbs buffer
    uint8_t* qcir_ptr = static_cast<uint8_t*>(qcir_binary.buffer);

    std::memcpy(qnn_custom_buffer_.data() + pos, qcir_ptr, qcir_fbs_size_);
    pos += qcir_fbs_size_;

    // tensor data
    std::memcpy(
        qnn_custom_buffer_.data() + pos, tensor_data.data(), tensor_size_);
  }
}

std::tuple<Error, uint32_t, uint64_t, void*, void*>
QnnQcirCustomProtocol::DeserializeQcirCustomBuffer(void* processed_data) {
  Error status = Error::Ok;
  uint8_t* ptr = static_cast<uint8_t*>(processed_data);
  size_t magic_number_proto_size = sizeof(magic_number_);
  uint8_t qcir_fbs_proto_size = sizeof(qcir_fbs_size_);
  uint8_t tensor_proto_size = sizeof(tensor_size_);

  uint32_t magic_number;
  std::memcpy(&magic_number, ptr, magic_number_proto_size);
  ptr += magic_number_proto_size;

  if (magic_number != magic_number_) {
    QNN_EXECUTORCH_LOG_INFO(
        "QnnQcirCustomProtocol expected magic number: 0x%x but get: 0x%x",
        magic_number_,
        magic_number);
    status = Error::Internal;
  }

  // Retrieve size of qcir.fbs
  uint32_t qcir_fbs_size;
  std::memcpy(&qcir_fbs_size, ptr, qcir_fbs_proto_size);
  ptr += qcir_fbs_proto_size;

  // Retrieve size of tensor
  uint64_t tensor_size;
  std::memcpy(&tensor_size, ptr, tensor_proto_size);
  ptr += tensor_proto_size;

  // Retrieve qcir.fbs pointer
  void* qcir_fbs_ptr = static_cast<void*>(ptr);
  ptr += qcir_fbs_size;

  // Retrieve tensor
  void* tensor_ptr = static_cast<void*>(ptr);

  return {status, qcir_fbs_size, tensor_size, qcir_fbs_ptr, tensor_ptr};
}

void QnnContextCustomProtocol::BuildContextCustomBuffer() {
  if (qnn_custom_buffer_.size() == 0) {
    signature_ =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();

    uint8_t magic_number_proto_size = sizeof(magic_number_);
    uint8_t binary_proto_size = sizeof(binary_size_);
    uint8_t signature_proto_size = sizeof(signature_);
    uint64_t buffer_size = magic_number_proto_size + signature_proto_size +
        binary_proto_size + binary_size_;
    qnn_custom_buffer_.resize(buffer_size, 0);

    size_t pos = 0;

    // magic number itself
    std::memcpy(
        qnn_custom_buffer_.data(), &magic_number_, magic_number_proto_size);
    pos += magic_number_proto_size;

    // signature itself
    std::memcpy(
        qnn_custom_buffer_.data() + pos, &signature_, signature_proto_size);
    pos += signature_proto_size;

    // size of context binary, should be 8 bytes
    // Binary itself won't be stored here. Refer to QnnCustomProtocol.h for more
    // info.
    std::memcpy(
        qnn_custom_buffer_.data() + pos, &binary_size_, binary_proto_size);
  }
}

void QnnContextCustomProtocol::BuildContextCustomBuffer(
    const QnnExecuTorchContextBinary& context_binary) {
  BuildContextCustomBuffer();
  uint64_t offset = GetContextBinaryOffset();
  std::memcpy(
      qnn_custom_buffer_.data() + offset,
      static_cast<uint8_t*>(context_binary.buffer),
      context_binary.nbytes);
}

std::tuple<Error, int64_t, uint64_t, void*>
QnnContextCustomProtocol::DeserializeContextCustomBuffer(void* processed_data) {
  Error status = Error::Ok;

  uint8_t* ptr = static_cast<uint8_t*>(processed_data);
  uint8_t magic_number_proto_size = sizeof(magic_number_);
  uint8_t binary_proto_size = sizeof(binary_size_);
  uint8_t signature_proto_size = sizeof(signature_);

  uint32_t magic_number;
  std::memcpy(&magic_number, ptr, magic_number_proto_size);
  ptr += magic_number_proto_size;

  if (magic_number != magic_number_) {
    QNN_EXECUTORCH_LOG_INFO(
        "QnnContextCustomProtocol expected magic number: 0x%x but get: 0x%x",
        magic_number_,
        magic_number);
    status = Error::Internal;
  }

  std::memcpy(&signature_, ptr, signature_proto_size);
  ptr += signature_proto_size;

  uint64_t binary_size;
  std::memcpy(&binary_size, ptr, binary_proto_size);
  ptr += binary_proto_size;

  return {status, signature_, binary_size, static_cast<void*>(ptr)};
}

uint64_t QnnContextCustomProtocol::GetContextBinaryOffset() {
  return sizeof(magic_number_) + sizeof(signature_) + sizeof(binary_size_);
}

} // namespace qnn
} // namespace backends
} // namespace executorch
