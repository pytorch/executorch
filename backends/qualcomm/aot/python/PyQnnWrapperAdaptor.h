/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/aot/wrappers/OpWrapper.h>
#include <executorch/backends/qualcomm/aot/wrappers/TensorWrapper.h>
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
namespace torch {
namespace executor {
namespace qnn {
class PyQnnOpWrapper {
 public:
  explicit PyQnnOpWrapper(
      const std::string& name,
      const std::string& package_name,
      const std::string& op_type) {
    op_wrapper_ = std::make_shared<OpWrapper>(name, package_name, op_type);
  }
  void AddInputTensors(
      const std::vector<std::shared_ptr<TensorWrapper>>& tensors) {
    op_wrapper_->AddInputTensors(tensors);
  }

  void AddOutputTensors(
      const std::vector<std::shared_ptr<TensorWrapper>>& tensors) {
    op_wrapper_->AddOutputTensors(tensors);
  }

  void AddTensorParam(
      const std::string& name,
      Qnn_DataType_t data_type,
      std::uint32_t rank,
      const std::vector<uint32_t>& dims,
      py::array& data,
      bool copy_data) {
    op_wrapper_->AddTensorParam(
        name, data_type, rank, dims.data(), data.data(), copy_data);
  }

  void AddScalarParam(
      const std::string& name,
      Qnn_DataType_t data_type,
      py::dict& attrData) {
    switch (data_type) {
      case Qnn_DataType_t::QNN_DATATYPE_INT_32:
        op_wrapper_->AddScalarParam(
            name, data_type, attrData["data"].cast<int32_t>());
        break;
      case Qnn_DataType_t::QNN_DATATYPE_INT_16:
        op_wrapper_->AddScalarParam(
            name, data_type, attrData["data"].cast<int16_t>());
        break;
      case Qnn_DataType_t::QNN_DATATYPE_INT_8:
        op_wrapper_->AddScalarParam(
            name, data_type, attrData["data"].cast<int8_t>());
        break;
      case Qnn_DataType_t::QNN_DATATYPE_UINT_32:
        op_wrapper_->AddScalarParam(
            name, data_type, attrData["data"].cast<uint32_t>());
        break;
      case Qnn_DataType_t::QNN_DATATYPE_UINT_16:
        op_wrapper_->AddScalarParam(
            name, data_type, attrData["data"].cast<uint16_t>());
        break;
      case Qnn_DataType_t::QNN_DATATYPE_UINT_8:
        op_wrapper_->AddScalarParam(
            name, data_type, attrData["data"].cast<uint8_t>());
        break;
      case Qnn_DataType_t::QNN_DATATYPE_FLOAT_32:
      case Qnn_DataType_t::QNN_DATATYPE_FLOAT_16:
        op_wrapper_->AddScalarParam(
            name, data_type, attrData["data"].cast<float>());
        break;
      case Qnn_DataType_t::QNN_DATATYPE_BOOL_8:
        op_wrapper_->AddScalarParam(
            name, data_type, attrData["data"].cast<bool>());
        break;
      default:
        QNN_EXECUTORCH_LOG_ERROR("tensor.v1.name: %d", data_type);
        break;
    }
  }
  std::shared_ptr<OpWrapper>& GetOpWrapper() {
    return op_wrapper_;
  }

 private:
  std::shared_ptr<OpWrapper> op_wrapper_;
};
} // namespace qnn
} // namespace executor
} // namespace torch
