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
namespace executorch {
namespace backends {
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

std::string GetScalarValue(const Qnn_Scalar_t& scalar) {
  switch (scalar.dataType) {
    case QNN_DATATYPE_FLOAT_32:
      return std::to_string(scalar.floatValue);
    case QNN_DATATYPE_FLOAT_64:
      return std::to_string(scalar.doubleValue);
    case QNN_DATATYPE_UINT_64:
      return std::to_string(scalar.uint64Value);
    case QNN_DATATYPE_INT_64:
      return std::to_string(scalar.int64Value);
    case QNN_DATATYPE_UINT_32:
      return std::to_string(scalar.uint32Value);
    case QNN_DATATYPE_INT_32:
      return std::to_string(scalar.int32Value);
    case QNN_DATATYPE_UINT_16:
      return std::to_string(scalar.uint16Value);
    case QNN_DATATYPE_INT_16:
      return std::to_string(scalar.int16Value);
    case QNN_DATATYPE_UINT_8:
      return std::to_string(scalar.uint8Value);
    case QNN_DATATYPE_INT_8:
      return std::to_string(scalar.int8Value);
    case QNN_DATATYPE_BOOL_8:
      return std::to_string(static_cast<int>(scalar.bool8Value));
    case QNN_DATATYPE_STRING:
      return std::string(scalar.stringValue);
    default:
      return "QNN_DATATYPE_UNDEFINED";
  }
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
           const std::string&>())
      .def(
          "GetInputTensors",
          &OpWrapper::GetInputTensors,
          "A function which gets input tensors")
      .def(
          "GetOutputTensors",
          &OpWrapper::GetOutputTensors,
          "A function which gets output tensors")
      .def("GetOpType", &OpWrapper::GetOpType, "A function which gets op type")
      .def("GetName", &OpWrapper::GetName, "A function which gets name")
      .def(
          "GetPackageName",
          &OpWrapper::GetPackageName,
          "A function which gets package name")
      .def(
          "GetParams", &OpWrapper::GetRawParams, "A function which gets params")
      // lambda function
      // python: op_wrapper.GetOpConfig()
      .def(
          "GetOpConfig",
          [](OpWrapper& self) {
            auto op_config = self.GetOpConfig();
            py::dict result;
            py::list params_list;
            py::list input_tensors_list;
            py::list output_tensors_list;
            result["version"] = op_config.version;
            result["name"] = op_config.v1.name;
            result["packageName"] = op_config.v1.packageName;
            result["typeName"] = op_config.v1.typeName;
            result["numOfParams"] = op_config.v1.numOfParams;
            for (size_t i = 0; i < op_config.v1.numOfParams; ++i) {
              params_list.append(op_config.v1.params[i]);
            }
            result["params"] = params_list;
            result["numOfInputs"] = op_config.v1.numOfInputs;
            for (size_t i = 0; i < op_config.v1.numOfInputs; ++i) {
              input_tensors_list.append(op_config.v1.inputTensors[i]);
            }
            result["inputTensors"] = input_tensors_list;
            result["numOfOutputs"] = op_config.v1.numOfOutputs;
            for (size_t i = 0; i < op_config.v1.numOfOutputs; ++i) {
              output_tensors_list.append(op_config.v1.outputTensors[i]);
            }
            result["outputTensors"] = output_tensors_list;
            return result;
          },
          "Get operator configuration");

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
      .def(py::init<float, int32_t>())
      .def_readonly("scale", &Qnn_ScaleOffset_t::scale)
      .def_readonly("offset", &Qnn_ScaleOffset_t::offset);

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

  py::class_<Qnn_OpConfig_t>(m, "Qnn_OpConfig")
      .def_readonly("version", &Qnn_OpConfig_t::version)
      // getter
      // python: op_wrapper.GetOpConfig().v1
      .def_property_readonly(
          "v1", [](const Qnn_OpConfig_t& config) -> const Qnn_OpConfigV1_t& {
            return config.v1;
          });

  py::enum_<Qnn_OpConfigVersion_t>(m, "Qnn_OpConfigVersion")
      .value("QNN_OPCONFIG_VERSION_1", QNN_OPCONFIG_VERSION_1)
      .value("QNN_OPCONFIG_VERSION_UNDEFINED", QNN_OPCONFIG_VERSION_UNDEFINED)
      .export_values();

  py::class_<Qnn_OpConfigV1_t>(m, "Qnn_OpConfigV1")
      .def_readonly("name", &Qnn_OpConfigV1_t::name)
      .def_readonly("packageName", &Qnn_OpConfigV1_t::packageName)
      .def_readonly("typeName", &Qnn_OpConfigV1_t::typeName)
      .def_readonly("numOfParams", &Qnn_OpConfigV1_t::numOfParams)
      .def_readonly("params", &Qnn_OpConfigV1_t::params)
      .def_readonly("numOfInputs", &Qnn_OpConfigV1_t::numOfInputs)
      .def_readonly("inputTensors", &Qnn_OpConfigV1_t::inputTensors)
      .def_readonly("numOfOutputs", &Qnn_OpConfigV1_t::numOfOutputs)
      .def_readonly("outputTensors", &Qnn_OpConfigV1_t::outputTensors);

  py::class_<Qnn_Param_t>(m, "Qnn_Param")
      .def_readonly("paramType", &Qnn_Param_t::paramType)
      .def_readonly("name", &Qnn_Param_t::name)
      .def_property_readonly(
          "scalarParam",
          [](const Qnn_Param_t& param) -> const Qnn_Scalar_t& {
            if (param.paramType == Qnn_ParamType_t::QNN_PARAMTYPE_SCALAR) {
              return param.scalarParam;
            }
            throw std::runtime_error("ParamType is not scalar.");
          })
      .def_property_readonly(
          "tensorParam", [](const Qnn_Param_t& param) -> const Qnn_Tensor_t& {
            if (param.paramType == Qnn_ParamType_t::QNN_PARAMTYPE_TENSOR) {
              return param.tensorParam;
            }
            throw std::runtime_error("ParamType is not tensor.");
          });

  py::enum_<Qnn_ParamType_t>(m, "Qnn_ParamType_t")
      .value("QNN_PARAMTYPE_SCALAR", Qnn_ParamType_t::QNN_PARAMTYPE_SCALAR)
      .value("QNN_PARAMTYPE_TENSOR", Qnn_ParamType_t::QNN_PARAMTYPE_TENSOR)
      .value(
          "QNN_PARAMTYPE_UNDEFINED", Qnn_ParamType_t::QNN_PARAMTYPE_UNDEFINED)
      .export_values();

  py::class_<Qnn_Scalar_t>(m, "Qnn_Scalar_t")
      .def_readonly("dataType", &Qnn_Scalar_t::dataType)
      .def("value", &GetScalarValue, "Get the value of the scalar as a string");

  py::class_<Qnn_Tensor_t>(m, "Qnn_Tensor_t")
      .def_readonly("version", &Qnn_Tensor_t::version)
      .def_property_readonly(
          "v1",
          [](Qnn_Tensor_t& t) -> Qnn_TensorV1_t& {
            if (t.version == QNN_TENSOR_VERSION_1) {
              return t.v1;
            }
            throw std::runtime_error("Tensor version is not V1.");
          })
      .def_property_readonly("v2", [](Qnn_Tensor_t& t) -> Qnn_TensorV2_t& {
        if (t.version == QNN_TENSOR_VERSION_2) {
          return t.v2;
        }
        throw std::runtime_error("Tensor version is not V2.");
      });

  py::enum_<Qnn_TensorVersion_t>(m, "Qnn_TensorVersion_t")
      .value("QNN_TENSOR_VERSION_1", Qnn_TensorVersion_t::QNN_TENSOR_VERSION_1)
      .value("QNN_TENSOR_VERSION_2", Qnn_TensorVersion_t::QNN_TENSOR_VERSION_2)
      .value(
          "QNN_TENSOR_VERSION_UNDEFINED",
          Qnn_TensorVersion_t::QNN_TENSOR_VERSION_UNDEFINED)
      .export_values();

  py::class_<Qnn_TensorV1_t>(m, "QnnTensorV1")
      .def_readonly("id", &Qnn_TensorV1_t::id)
      .def_readonly("name", &Qnn_TensorV1_t::name)
      .def_readonly("type", &Qnn_TensorV1_t::type)
      .def_readonly("dataFormat", &Qnn_TensorV1_t::dataFormat)
      .def_readonly("dataType", &Qnn_TensorV1_t::dataType)
      .def_readonly("quantizeParams", &Qnn_TensorV1_t::quantizeParams)
      .def_readonly("rank", &Qnn_TensorV1_t::rank)
      // change dimensions pointer to vector(begin to rank)
      .def_property_readonly(
          "dimensions",
          [](const Qnn_TensorV1_t& t) {
            return std::vector<uint32_t>(t.dimensions, t.dimensions + t.rank);
          })
      .def_readonly("memType", &Qnn_TensorV1_t::memType);

  py::enum_<Qnn_TensorMemType_t>(m, "Qnn_TensorMemType_t")
      .value(
          "QNN_TENSORMEMTYPE_RAW", Qnn_TensorMemType_t::QNN_TENSORMEMTYPE_RAW)
      .value(
          "QNN_TENSORMEMTYPE_MEMHANDLE",
          Qnn_TensorMemType_t::QNN_TENSORMEMTYPE_MEMHANDLE)
      .value(
          "QNN_TENSORMEMTYPE_UNDEFINED",
          Qnn_TensorMemType_t::QNN_TENSORMEMTYPE_UNDEFINED)
      .export_values();

  py::class_<Qnn_QuantizeParams_t>(m, "QnnQuantizeParams")
      .def_readonly(
          "encodingDefinition", &Qnn_QuantizeParams_t::encodingDefinition)
      .def_readonly(
          "quantizationEncoding", &Qnn_QuantizeParams_t::quantizationEncoding)
      .def_property_readonly(
          "scaleOffsetEncoding",
          [](const Qnn_QuantizeParams_t& qp) {
            if (qp.quantizationEncoding ==
                QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
              return qp.scaleOffsetEncoding;
            }
            throw std::runtime_error(
                "Invalid quantization encoding type for scaleOffsetEncoding.");
          })
      .def_property_readonly(
          "axisScaleOffsetEncoding", [](const Qnn_QuantizeParams_t& qp) {
            if (qp.quantizationEncoding ==
                QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
              return qp.axisScaleOffsetEncoding;
            }
            throw std::runtime_error(
                "Invalid quantization encoding type for axisScaleOffsetEncoding.");
          });

  py::enum_<Qnn_Definition_t>(m, "QnnDefinition")
      .value(
          "QNN_DEFINITION_IMPL_GENERATED",
          Qnn_Definition_t::QNN_DEFINITION_IMPL_GENERATED)
      .value("QNN_DEFINITION_DEFINED", Qnn_Definition_t::QNN_DEFINITION_DEFINED)
      .value(
          "QNN_DEFINITION_UNDEFINED",
          Qnn_Definition_t::QNN_DEFINITION_UNDEFINED)
      .export_values();

  py::class_<Qnn_AxisScaleOffset_t>(m, "QnnAxisScaleOffset")
      .def_readonly("axis", &Qnn_AxisScaleOffset_t::axis)
      .def_readonly("numScaleOffsets", &Qnn_AxisScaleOffset_t::numScaleOffsets)
      .def_property_readonly(
          "scaleOffset", [](const Qnn_AxisScaleOffset_t& aso) {
            return std::vector<Qnn_ScaleOffset_t>(
                aso.scaleOffset, aso.scaleOffset + aso.numScaleOffsets);
          });
  // op_wrapper.GetParams() get std::vector<ParamWrapper*>
}
} // namespace qnn
} // namespace backends
} // namespace executorch
