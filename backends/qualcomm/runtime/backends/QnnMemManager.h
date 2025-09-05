/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <executorch/backends/qualcomm/aot/wrappers/TensorWrapper.h>
#include <executorch/backends/qualcomm/runtime/SharedBuffer.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnContextCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnImplementation.h>
#include <unordered_map>
#include "HTP/QnnHtpMem.h"

namespace executorch {
namespace backends {
namespace qnn {

class QnnMemManager {
 public:
  explicit QnnMemManager(
      const QnnImplementation& implementation,
      QnnContext* context,
      QnnExecuTorchLogLevel log_level)
      : implementation_(implementation),
        context_(context),
        log_level_(log_level) {}
  ~QnnMemManager() {
    DeRegisterMem();
  }

  executorch::runtime::Error RegisterIonMem(
      const std::shared_ptr<TensorWrapper>& tensor_wrapper,
      int32_t mem_fd,
      void* mem_ptr);

  executorch::runtime::Error RegisterCustomMem(
      const std::shared_ptr<TensorWrapper>& tensor_wrapper,
      int32_t mem_fd,
      void* mem_ptr,
      void* unaligned_custom_mem_base,
      size_t total_custom_mem_size,
      size_t tensor_offset,
      const CustomMemTensorInfo& info);

  // Pre-register custom mem handle from SharedBuffer. Bring forward the
  // memHandle creating time from execution to initialization.
  executorch::runtime::Error PreRegisterCustomMemHandle(
      int32_t mem_fd,
      void* unaligned_custom_mem_base,
      size_t total_custom_mem_size,
      size_t tensor_offset,
      const CustomMemTensorInfo& info);

  bool IsRegistered(Qnn_MemHandle_t handle, void* mem_ptr);

  void* GetPreRegisteredHandle(const CustomMemTensorInfo& info);

  executorch::runtime::Error SetMemHandle(
      const std::shared_ptr<TensorWrapper>& tensor_wrapper,
      void* mem_ptr,
      Qnn_MemHandle_t handle);

 private:
  void DeRegisterMem();

  const QnnImplementation& implementation_;
  QnnContext* context_;
  QnnExecuTorchLogLevel log_level_;
  // Store the registered Qnn_MemHandle_t for de-registration
  std::unordered_map<Qnn_MemHandle_t, void*> registered_map_;
  // Store the pre-registered custom mem handles
  std::unordered_map<CustomMemTensorInfo, void*> pre_registered_handles_;
  std::unordered_map<executorch::aten::ScalarType, Qnn_DataType_t>
      scalar_type_to_qnn_dtype_ = {
          {executorch::aten::ScalarType::Int,
           Qnn_DataType_t::QNN_DATATYPE_INT_32},
          {executorch::aten::ScalarType::Float,
           Qnn_DataType_t::QNN_DATATYPE_FLOAT_32},
          {executorch::aten::ScalarType::Char,
           Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_8},
          {executorch::aten::ScalarType::Short,
           Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_16},
          {executorch::aten::ScalarType::Byte,
           Qnn_DataType_t::QNN_DATATYPE_UFIXED_POINT_8},
          {executorch::aten::ScalarType::UInt16,
           Qnn_DataType_t::QNN_DATATYPE_UFIXED_POINT_16},
  };
};
} // namespace qnn
} // namespace backends
} // namespace executorch
