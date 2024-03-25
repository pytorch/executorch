/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/aot/wrappers/TensorWrapper.h>

#include <atomic>
#include <cstring>
#include <limits>
#include <numeric>
namespace torch {
namespace executor {
namespace qnn {
std::uint32_t GetDataTypeSize(Qnn_DataType_t data_type) {
  std::uint32_t size = 0;

  switch (data_type) {
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_BOOL_8:
      size = sizeof(std::uint8_t);
      break;
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      size = sizeof(std::uint16_t);
      break;
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32:
      size = sizeof(float);
      break;
    case QNN_DATATYPE_INT_64:
    case QNN_DATATYPE_UINT_64:
      size = sizeof(std::uint64_t);
      break;
    case QNN_DATATYPE_UNDEFINED:
    default:
      size = 0;
  }

  return size;
}

std::atomic<std::uint32_t> intermediate_tensor_id{
    std::numeric_limits<std::uint32_t>::max()};

std::uint32_t CreateIntermediateTensorId() {
  return --intermediate_tensor_id;
}

TensorWrapper::TensorWrapper(
    const std::string& tensor_name,
    Qnn_TensorType_t tensor_type,
    Qnn_DataType_t data_type,
    std::unique_ptr<QuantizeParamsWrapper> quantize_param_wrapper,
    std::uint32_t rank,
    const std::uint32_t dims[],
    std::uint32_t bytes,
    const void* data,
    bool copy_data)
    : qnn_tensor_name_(tensor_name),
      quantize_param_wrapper_(std::move(quantize_param_wrapper)),
      dims_(dims, dims + rank),
      bytes_(bytes),
      owned_data_(nullptr) {
  // "version" is the only exception that we don't need QNN_VER_PTR wrapper.
  tensor_.version = QNN_TENSOR_VERSION_1;

  // Don't assign .id because it's an output field.
  QNN_VER_PTR(tensor_)->name = qnn_tensor_name_.c_str();
  QNN_VER_PTR(tensor_)->dimensions = dims_.data();
  QNN_VER_PTR(tensor_)->type = tensor_type;
  QNN_VER_PTR(tensor_)->dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  QNN_VER_PTR(tensor_)->dataType = data_type;
  QNN_VER_PTR(tensor_)->quantizeParams =
      quantize_param_wrapper_->CreateQuantizeParams();
  QNN_VER_PTR(tensor_)->rank = rank;
  QNN_VER_PTR(tensor_)->memType = QNN_TENSORMEMTYPE_RAW;

  if (data != nullptr) {
    QNN_VER_PTR(tensor_)->clientBuf.dataSize = bytes;

    if (copy_data) {
      owned_data_ = std::make_unique<char[]>(bytes);
      const char* src_data = static_cast<const char*>(data);
      std::memcpy(owned_data_.get(), src_data, bytes);
      QNN_VER_PTR(tensor_)->clientBuf.data = owned_data_.get();
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      QNN_VER_PTR(tensor_)->clientBuf.data = const_cast<void*>(data);
    }
  }
}

Error TensorWrapper::FillDataBuffer(const void* data, bool copy_data) {
  if (data != nullptr) {
    QNN_VER_PTR(tensor_)->memType = QNN_TENSORMEMTYPE_RAW;
    QNN_VER_PTR(tensor_)->clientBuf.dataSize = bytes_;
    if (copy_data) {
      owned_data_ = std::make_unique<char[]>(bytes_);
      const char* src_data = static_cast<const char*>(data);
      std::memcpy(owned_data_.get(), src_data, bytes_);
      QNN_VER_PTR(tensor_)->clientBuf.data = owned_data_.get();
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      QNN_VER_PTR(tensor_)->clientBuf.data = const_cast<void*>(data);
    }
  } else {
    QNN_EXECUTORCH_LOG_WARN("Data pointer is nullptr");
  }
  return Error::Ok;
}

Error TensorWrapper::AllocateDataBuffer() {
  char* static_data_buffer = new (std::nothrow) char[bytes_]; // NOLINT
  if (static_data_buffer == nullptr) {
    return Error::Internal;
  }
  owned_data_ = std::unique_ptr<char[]>(static_data_buffer);
  QNN_VER_PTR(tensor_)->memType = QNN_TENSORMEMTYPE_RAW;
  QNN_VER_PTR(tensor_)->clientBuf.dataSize = bytes_;
  QNN_VER_PTR(tensor_)->clientBuf.data = owned_data_.get();

  return Error::Ok;
}

void TensorWrapper::UpdateQnnTensorMeta(const Qnn_Tensor_t& tensor_src) {
  QNN_VER_PTR(tensor_)->id = QNN_VER_PTR(tensor_src)->id;
}

Error TensorWrapper::SetName(const std::string& name) {
  qnn_tensor_name_ = name;
  QNN_VER_PTR(tensor_)->name = qnn_tensor_name_.c_str();
  return Error::Ok;
}

Error TensorWrapper::SetMemHandle(Qnn_MemHandle_t mem_handle) {
  QNN_VER_PTR(tensor_)->memType = QNN_TENSORMEMTYPE_MEMHANDLE;
  QNN_VER_PTR(tensor_)->memHandle = mem_handle;
  return Error::Ok;
}

// base function for Create TensorWrapper
std::shared_ptr<TensorWrapper> CreateTensorWrapper(
    const std::string& tensor_name,
    Qnn_TensorType_t tensor_type,
    Qnn_DataType_t data_type,
    std::unique_ptr<QuantizeParamsWrapper> quantize_param_wrapper,
    std::uint32_t rank,
    const std::uint32_t dims[],
    std::uint32_t bytes,
    const void* data,
    bool copy_data) {
  if (bytes == 0) {
    bytes = std::accumulate(
        dims, dims + rank, GetDataTypeSize(data_type), std::multiplies<>());
  }
  return std::make_shared<TensorWrapper>(
      tensor_name,
      tensor_type,
      data_type,
      std::move(quantize_param_wrapper),
      rank,
      dims,
      bytes,
      data,
      copy_data);
}

std::shared_ptr<TensorWrapper> CreateTensorWrapper(
    Qnn_TensorType_t tensor_type,
    Qnn_DataType_t data_type,
    std::unique_ptr<QuantizeParamsWrapper> quantize_param_wrapper,
    std::uint32_t rank,
    const std::uint32_t dims[],
    std::uint32_t bytes,
    const void* data,
    bool copy_data) {
  return CreateTensorWrapper(
      std::to_string(CreateIntermediateTensorId()),
      tensor_type,
      data_type,
      std::move(quantize_param_wrapper),
      rank,
      dims,
      bytes,
      data,
      copy_data);
}

// Factory functions to create TensorWrappers
std::shared_ptr<TensorWrapper> CreateTensorWrapper(const Qnn_Tensor_t& tensor) {
  return CreateTensorWrapper(
      std::string(QNN_VER_PTR(tensor)->name),
      QNN_VER_PTR(tensor)->type,
      QNN_VER_PTR(tensor)->dataType,
      CreateQuantizationParamWrapper(QNN_VER_PTR(tensor)->quantizeParams),
      QNN_VER_PTR(tensor)->rank,
      QNN_VER_PTR(tensor)->dimensions,
      QNN_VER_PTR(tensor)->clientBuf.dataSize,
      QNN_VER_PTR(tensor)->clientBuf.data);
}
} // namespace qnn
} // namespace executor
} // namespace torch
