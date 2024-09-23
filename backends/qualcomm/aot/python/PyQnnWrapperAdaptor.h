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
        QNN_EXECUTORCH_LOG_ERROR(
            "%s has invalid data type: %d", name.c_str(), data_type);
        break;
    }
  }
  std::shared_ptr<OpWrapper>& GetOpWrapper() {
    return op_wrapper_;
  }

 private:
  std::shared_ptr<OpWrapper> op_wrapper_;
};

class PyQnnTensorWrapper {
 public:
  explicit PyQnnTensorWrapper(const std::shared_ptr<TensorWrapper>& wrapper) {
    tensor_wrapper_ = wrapper;
  }
  struct EncodingData {
    float scale;
    int32_t offset;
  };
  struct Encoding {
    py::array_t<EncodingData> data;
    int32_t axis;
  };

  py::array_t<std::uint32_t> GetDims() {
    std::uint32_t* dim = tensor_wrapper_->GetDims();
    size_t shape[1]{tensor_wrapper_->GetRank()};
    size_t stride[1]{sizeof(std::uint32_t)};
    auto ret = py::array_t<std::uint32_t>(shape, stride);
    auto view = ret.mutable_unchecked<1>();
    for (int i = 0; i < ret.shape(0); ++i) {
      view(i) = dim[i];
    }
    return ret;
  }
  std::string GetName() {
    return tensor_wrapper_->GetName();
  }
  Qnn_DataType_t GetDataType() {
    return tensor_wrapper_->GetDataType();
  }
  Encoding GetEncodings() {
    auto q_param = tensor_wrapper_->GetQuantizeParams();
    size_t stride[1]{sizeof(EncodingData)};

    switch (q_param.quantizationEncoding) {
      case QNN_QUANTIZATION_ENCODING_SCALE_OFFSET: {
        Qnn_ScaleOffset_t data = q_param.scaleOffsetEncoding;
        size_t shape[1]{1};
        auto enc_data = py::array_t<EncodingData>(shape, stride);
        auto view = enc_data.mutable_unchecked<1>();
        view(0) = {data.scale, data.offset};
        return {enc_data, -1};
      }
      case QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET: {
        Qnn_AxisScaleOffset_t data = q_param.axisScaleOffsetEncoding;
        size_t shape[1]{data.numScaleOffsets};
        auto enc_data = py::array_t<EncodingData>(shape, stride);
        auto view = enc_data.mutable_unchecked<1>();
        for (int i = 0; i < enc_data.shape(0); ++i) {
          view(i) = {data.scaleOffset[i].scale, data.scaleOffset[i].offset};
        }
        return {enc_data, data.axis};
      }
      case QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET: {
        Qnn_BwScaleOffset_t data = q_param.bwScaleOffsetEncoding;
        size_t shape[1]{1};
        auto enc_data = py::array_t<EncodingData>(shape, stride);
        auto view = enc_data.mutable_unchecked<1>();
        view(0) = {data.scale, data.offset};
        return {enc_data, -1};
      }
      case QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET: {
        Qnn_BwAxisScaleOffset_t data = q_param.bwAxisScaleOffsetEncoding;
        size_t shape[1]{data.numElements};
        auto enc_data = py::array_t<EncodingData>(shape, stride);
        auto view = enc_data.mutable_unchecked<1>();
        for (int i = 0; i < enc_data.shape(0); ++i) {
          view(i) = {data.scales[i], data.offsets[i]};
        }
        return {enc_data, data.axis};
      }
      default:
        QNN_EXECUTORCH_LOG_WARN(
            "%s QNN_QUANTIZATION_ENCODING_UNDEFINED detected",
            GetName().c_str());
        break;
    }
    return {};
  }

 private:
  std::shared_ptr<TensorWrapper> tensor_wrapper_;
};
} // namespace qnn
} // namespace executor
} // namespace torch
