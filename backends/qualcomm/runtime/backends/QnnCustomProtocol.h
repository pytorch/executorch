/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/runtime/core/error.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

namespace executorch {
namespace backends {
namespace qnn {

using executorch::runtime::Error;

// We have 2 kinds of protocol here: custom_qcir_protocol,
// custom_context_protocol. We need this class due to limitation of 32bits
// flatbuffer. Since larger models can exceed the maximum size for 32bits
// flatbuffer, we need to define our own protocol and store some information
// outside of the flatbuffer. The magic number helps determine if we are getting
// the correct custom protocol buffer and differentiate custom_qcir_protocol
// from custom_context_protocol.
class QnnCustomProtocol {
 public:
  QnnCustomProtocol() {}

  // Get a pair that holds pointer pointing to the start of custom buffer and
  // the size of the custom buffer.
  std::pair<void*, uint64_t> GetCustomProtocolBuffer() {
    return {
        static_cast<void*>(qnn_custom_buffer_.data()),
        qnn_custom_buffer_.size()};
  }

 protected:
  std::vector<uint8_t> qnn_custom_buffer_;
};

// For custom_qcir_protocol, we expect the following format:
//
// ------------------------------
// | qcir magic number (4 bytes)|
// ------------------------------
// | qcir.fbs size (4 bytes)    |
// ------------------------------
// | tensor size (8 bytes)      |
// ------------------------------
// | qcir.fbs (flatbuffer)      |
// ------------------------------
// | tensor.data                |
// ------------------------------
class QnnQcirCustomProtocol : public QnnCustomProtocol {
 public:
  // Constructor for Serialize
  QnnQcirCustomProtocol(uint32_t qcir_fbs_size, uint64_t tensor_size)
      : QnnCustomProtocol(),
        qcir_fbs_size_(qcir_fbs_size),
        tensor_size_(tensor_size) {}

  // Constructor for Deserialize
  QnnQcirCustomProtocol() : QnnCustomProtocol() {}

  void BuildQcirCustomBuffer(
      const QnnExecuTorchContextBinary& qcir_binary,
      const std::vector<uint8_t>& tensor_data);
  // Return a tuple with 5 elements:
  // 1) Error: Status of whether deserializing is successful.
  // 2) uint32_t: Size of qcir fbs
  // 3) uint64_t: Size of tensor
  // 4) void*: Pointer pointing to the start of qcir fbs
  // 5) void*: Pointer pointing to the start of tensor
  std::tuple<Error, uint32_t, uint64_t, void*, void*>
  DeserializeQcirCustomBuffer(void* processed_data);

 private:
  static constexpr uint32_t magic_number_ = 0x1234ABCD;
  uint32_t qcir_fbs_size_{0};
  uint64_t tensor_size_{0};
};

// For custom context binary protocol, we expect the following format:
//
// ---------------------------------
// | magic number (4 bytes)        |
// ---------------------------------
// | signature (8 bytes)           |
// ---------------------------------
// | context_binary_size (8 bytes) |
// ---------------------------------
// | context_binary.data           |
// ---------------------------------
class QnnContextCustomProtocol : public QnnCustomProtocol {
 public:
  // Constructor for Serialize
  QnnContextCustomProtocol(uint64_t binary_size)
      : QnnCustomProtocol(), binary_size_(binary_size) {}

  // Constructor for Deserialize
  QnnContextCustomProtocol() : QnnCustomProtocol() {}

  // Please note that this function will only initialize the required memory
  // space and fill in all meta data except for context_binary.data. Users will
  // need to handle context_binary.data themselves. This is because QNN-provided
  // functions, such as qnn_context_get_binary(), ask for a memory address
  // to store data and will fill it in for us.
  void BuildContextCustomBuffer();
  // Use this function if you already have context_binary ahead of time.
  void BuildContextCustomBuffer(const QnnExecuTorchContextBinary& qcir_binary);
  // Return a tuple with 4 elements:
  // 1) Error: Status of whether deserializing is successful.
  // 2) int64_t: Graph signature
  // 3) uint64_t: Size of the context binary
  // 4) void*: Pointer pointing to the start of context_binary
  std::tuple<Error, int64_t, uint64_t, void*> DeserializeContextCustomBuffer(
      void* processed_data);
  uint64_t GetContextBinaryOffset();

 private:
  static constexpr uint32_t magic_number_ = 0x5678ABCD;
  int64_t signature_{0};
  uint64_t binary_size_{0};
};

} // namespace qnn
} // namespace backends
} // namespace executorch
