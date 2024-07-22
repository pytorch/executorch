/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/aot/python/PyQnnWrapperAdaptor.h>
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

namespace py = pybind11;
namespace torch {
namespace executor {
namespace qnn {
std::unique_ptr<QuantizeParamsWrapper> CreateQuantizationParamWrapper(
    const Qnn_QuantizationEncoding_t& encoding,
    py::dict& quant_info) {
  std::unique_ptr<QuantizeParamsWrapper> quantize_param_wrapper;
  if (encoding == QNN_QUANTIZATION_ENCODING_UNDEFINED) {
    quantize_param_wrapper = std::make_unique<UndefinedQuantizeParamsWrapper>();
  } else if (encoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    int32_t axis = quant_info["axis"].cast<int32_t>();
    std::vector<Qnn_ScaleOffset_t> scale_offset =
        quant_info["scale_offset"].cast<std::vector<Qnn_ScaleOffset_t>>();

    quantize_param_wrapper =
        std::make_unique<AxisScaleOffsetQuantizeParamsWrapper>(
            axis, scale_offset);
  } else if (encoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
    uint32_t bitwidth = quant_info["bitwidth"].cast<uint32_t>();
    int32_t axis = quant_info["axis"].cast<int32_t>();
    std::vector<Qnn_ScaleOffset_t> scale_offset =
        quant_info["scale_offset"].cast<std::vector<Qnn_ScaleOffset_t>>();
    uint32_t num_elements = scale_offset.size();
    std::vector<float> scales;
    std::vector<int32_t> offsets;
    for (const auto& scale_offset : scale_offset) {
      scales.push_back(scale_offset.scale);
      offsets.push_back(scale_offset.offset);
    }
    quantize_param_wrapper =
        std::make_unique<BwAxisScaleOffsetQuantizeParamsWrapper>(
            bitwidth, axis, num_elements, scales, offsets);
  } else if (encoding == QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET) {
    uint32_t bitwidth = quant_info["bitwidth"].cast<uint32_t>();
    float scale = quant_info["scale"].cast<float>();
    int32_t offset = quant_info["offset"].cast<int32_t>();
    quantize_param_wrapper =
        std::make_unique<BwScaleOffsetQuantizeParamsWrapper>(
            bitwidth, scale, offset);
  } else if (encoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    float scale = quant_info["scale"].cast<float>();
    int32_t offset = quant_info["offset"].cast<int32_t>();
    quantize_param_wrapper =
        std::make_unique<ScaleOffsetQuantizeParamsWrapper>(scale, offset);
  } else {
    QNN_EXECUTORCH_LOG_ERROR(
        "Unknown the encoding of quantization: %d", encoding);
  }
  return quantize_param_wrapper;
}

std::shared_ptr<TensorWrapper> CreateTensorWrapper(
    const std::string& tensor_name,
    Qnn_TensorType_t tensor_type,
    Qnn_DataType_t data_type,
    const Qnn_QuantizationEncoding_t& encoding,
    py::dict& quant_info,
    std::uint32_t rank,
    const std::vector<uint32_t>& dims,
    py::array& data,
    bool copy_data) {
  std::unique_ptr<QuantizeParamsWrapper> quantize_param_wrapper =
      CreateQuantizationParamWrapper(encoding, quant_info);

  if (data.size() == 0) {
    return CreateTensorWrapper(
        tensor_name,
        tensor_type,
        data_type,
        std::move(quantize_param_wrapper),
        rank,
        dims.data(),
        0,
        nullptr,
        copy_data);
  }
  return CreateTensorWrapper(
      tensor_name,
      tensor_type,
      data_type,
      std::move(quantize_param_wrapper),
      rank,
      dims.data(),
      0,
      data.data(),
      copy_data);
}

PYBIND11_MODULE(PyQnnWrapperAdaptor, m) {
  PYBIND11_NUMPY_DTYPE(PyQnnTensorWrapper::EncodingData, scale, offset);

  py::enum_<Qnn_TensorType_t>(m, "Qnn_TensorType_t")
      .value(
          "QNN_TENSOR_TYPE_APP_WRITE",
          Qnn_TensorType_t::QNN_TENSOR_TYPE_APP_WRITE)
      .value(
          "QNN_TENSOR_TYPE_APP_READ",
          Qnn_TensorType_t::QNN_TENSOR_TYPE_APP_READ)
      .value(
          "QNN_TENSOR_TYPE_APP_READWRITE",
          Qnn_TensorType_t::QNN_TENSOR_TYPE_APP_READWRITE)
      .value("QNN_TENSOR_TYPE_NATIVE", Qnn_TensorType_t::QNN_TENSOR_TYPE_NATIVE)
      .value("QNN_TENSOR_TYPE_STATIC", Qnn_TensorType_t::QNN_TENSOR_TYPE_STATIC)
      .value("QNN_TENSOR_TYPE_NULL", Qnn_TensorType_t::QNN_TENSOR_TYPE_NULL)
      .value(
          "QNN_TENSOR_TYPE_UNDEFINED",
          Qnn_TensorType_t::QNN_TENSOR_TYPE_UNDEFINED)
      .export_values();

  py::enum_<Qnn_DataType_t>(m, "Qnn_DataType_t")
      .value("QNN_DATATYPE_INT_8", Qnn_DataType_t::QNN_DATATYPE_INT_8)
      .value("QNN_DATATYPE_INT_16", Qnn_DataType_t::QNN_DATATYPE_INT_16)
      .value("QNN_DATATYPE_INT_32", Qnn_DataType_t::QNN_DATATYPE_INT_32)
      .value("QNN_DATATYPE_INT_64", Qnn_DataType_t::QNN_DATATYPE_INT_64)
      .value("QNN_DATATYPE_UINT_8", Qnn_DataType_t::QNN_DATATYPE_UINT_8)
      .value("QNN_DATATYPE_UINT_16", Qnn_DataType_t::QNN_DATATYPE_UINT_16)
      .value("QNN_DATATYPE_UINT_32", Qnn_DataType_t::QNN_DATATYPE_UINT_32)
      .value("QNN_DATATYPE_UINT_64", Qnn_DataType_t::QNN_DATATYPE_UINT_64)
      .value("QNN_DATATYPE_FLOAT_16", Qnn_DataType_t::QNN_DATATYPE_FLOAT_16)
      .value("QNN_DATATYPE_FLOAT_32", Qnn_DataType_t::QNN_DATATYPE_FLOAT_32)
      .value(
          "QNN_DATATYPE_SFIXED_POINT_8",
          Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_8)
      .value(
          "QNN_DATATYPE_SFIXED_POINT_16",
          Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_16)
      .value(
          "QNN_DATATYPE_SFIXED_POINT_32",
          Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_32)
      .value(
          "QNN_DATATYPE_UFIXED_POINT_8",
          Qnn_DataType_t::QNN_DATATYPE_UFIXED_POINT_8)
      .value(
          "QNN_DATATYPE_UFIXED_POINT_16",
          Qnn_DataType_t::QNN_DATATYPE_UFIXED_POINT_16)
      .value(
          "QNN_DATATYPE_UFIXED_POINT_32",
          Qnn_DataType_t::QNN_DATATYPE_UFIXED_POINT_32)
      .value("QNN_DATATYPE_BOOL_8", Qnn_DataType_t::QNN_DATATYPE_BOOL_8)
      .value("QNN_DATATYPE_UNDEFINED", Qnn_DataType_t::QNN_DATATYPE_UNDEFINED)
      .export_values();

  py::enum_<Qnn_QuantizationEncoding_t>(m, "Qnn_QuantizationEncoding_t")
      .value(
          "QNN_QUANTIZATION_ENCODING_UNDEFINED",
          Qnn_QuantizationEncoding_t::QNN_QUANTIZATION_ENCODING_UNDEFINED)
      .value(
          "QNN_QUANTIZATION_ENCODING_SCALE_OFFSET",
          Qnn_QuantizationEncoding_t::QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
      .value(
          "QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET",
          Qnn_QuantizationEncoding_t::
              QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET)
      .value(
          "QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET",
          Qnn_QuantizationEncoding_t::QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET)
      .value(
          "QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET",
          Qnn_QuantizationEncoding_t::
              QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET)
      .export_values();
  py::class_<OpWrapper, std::shared_ptr<OpWrapper>>(m, "OpWrapper")
      .def(py::init<
           const std::string&,
           const std::string&,
           const std::string&>());

  py::class_<TensorWrapper, std::shared_ptr<TensorWrapper>>(m, "TensorWrapper")
      .def(py::init(py::overload_cast<
                    const std::string&,
                    Qnn_TensorType_t,
                    Qnn_DataType_t,
                    const Qnn_QuantizationEncoding_t&,
                    py::dict&,
                    std::uint32_t,
                    const std::vector<uint32_t>&,
                    py::array&,
                    bool>(&CreateTensorWrapper)));

  py::class_<QuantizeParamsWrapper>(m, "QuantizeParamsWrapper");

  py::class_<Qnn_ScaleOffset_t>(m, "Qnn_ScaleOffset_t")
      .def(py::init<float, int32_t>());

  py::class_<PyQnnOpWrapper, std::shared_ptr<PyQnnOpWrapper>>(
      m, "PyQnnOpWrapper")
      .def(py::init<
           const std::string&,
           const std::string&,
           const std::string&>())
      .def(
          "AddInputTensors",
          &PyQnnOpWrapper::AddInputTensors,
          "A function which add input tensor wrapper into op wrapper",
          py::arg("tensors"))
      .def(
          "AddOutputTensors",
          &PyQnnOpWrapper::AddOutputTensors,
          "A function which add output tensor wrapper into op wrapper",
          py::arg("tensors"))
      .def(
          "AddTensorParam",
          &PyQnnOpWrapper::AddTensorParam,
          "A function which add tensor parameter into op wrapper",
          py::arg("name"),
          py::arg("data_type"),
          py::arg("rank"),
          py::arg("dims"),
          py::arg("data"),
          py::arg("copy_data"))
      .def(
          "AddScalarParam",
          &PyQnnOpWrapper::AddScalarParam,
          "A function which add scalar parameter into op wrapper",
          py::arg("name"),
          py::arg("data_type"),
          py::arg("attrData"))
      .def(
          "GetOpWrapper",
          &PyQnnOpWrapper::GetOpWrapper,
          "A function which get op wrapper");

  py::class_<PyQnnTensorWrapper::Encoding>(m, "Encoding")
      .def_readonly("data", &PyQnnTensorWrapper::Encoding::data)
      .def_readonly("axis", &PyQnnTensorWrapper::Encoding::axis);

  py::class_<PyQnnTensorWrapper, std::shared_ptr<PyQnnTensorWrapper>>(
      m, "PyQnnTensorWrapper")
      .def(py::init<const std::shared_ptr<TensorWrapper>&>())
      .def("GetDims", &PyQnnTensorWrapper::GetDims)
      .def("GetDataType", &PyQnnTensorWrapper::GetDataType)
      .def("GetName", &PyQnnTensorWrapper::GetName)
      .def("GetEncodings", &PyQnnTensorWrapper::GetEncodings);
}
} // namespace qnn
} // namespace executor
} // namespace torch
