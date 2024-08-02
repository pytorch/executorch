/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/aot/ir/qcir_utils.h>
#include <executorch/backends/qualcomm/aot/wrappers/TensorWrapper.h>

#include <unordered_map>

namespace torch {
namespace executor {
namespace qnn {

qcir::TensorType ToTensorType(Qnn_TensorType_t type) {
  static const std::unordered_map<Qnn_TensorType_t, qcir::TensorType> type_map{
      {QNN_TENSOR_TYPE_APP_WRITE, qcir::TensorType::WRITE},
      {QNN_TENSOR_TYPE_APP_READ, qcir::TensorType::READ},
      {QNN_TENSOR_TYPE_APP_READWRITE, qcir::TensorType::READWRITE},
      {QNN_TENSOR_TYPE_NATIVE, qcir::TensorType::NATIVE},
      {QNN_TENSOR_TYPE_STATIC, qcir::TensorType::STATIC},
      {QNN_TENSOR_TYPE_NULL, qcir::TensorType::OPTIONAL},
      {QNN_TENSOR_TYPE_UNDEFINED, qcir::TensorType::UNDEFINED},
  };
  return type_map.at(type);
}

Qnn_TensorType_t ToTensorType(qcir::TensorType type) {
  static const std::unordered_map<qcir::TensorType, Qnn_TensorType_t> type_map{
      {qcir::TensorType::WRITE, QNN_TENSOR_TYPE_APP_WRITE},
      {qcir::TensorType::READ, QNN_TENSOR_TYPE_APP_READ},
      {qcir::TensorType::READWRITE, QNN_TENSOR_TYPE_APP_READWRITE},
      {qcir::TensorType::NATIVE, QNN_TENSOR_TYPE_NATIVE},
      {qcir::TensorType::STATIC, QNN_TENSOR_TYPE_STATIC},
      {qcir::TensorType::OPTIONAL, QNN_TENSOR_TYPE_NULL},
      {qcir::TensorType::UNDEFINED, QNN_TENSOR_TYPE_UNDEFINED},
  };
  return type_map.at(type);
}

// TODO: enable commented type by QNN version control
qcir::DataType ToDataType(Qnn_DataType_t type) {
  static const std::unordered_map<Qnn_DataType_t, qcir::DataType> type_map{
      {QNN_DATATYPE_INT_8, qcir::DataType::INT8},
      {QNN_DATATYPE_INT_16, qcir::DataType::INT16},
      {QNN_DATATYPE_INT_32, qcir::DataType::INT32},
      {QNN_DATATYPE_INT_64, qcir::DataType::INT64},
      {QNN_DATATYPE_UINT_8, qcir::DataType::UINT8},
      {QNN_DATATYPE_UINT_16, qcir::DataType::UINT16},
      {QNN_DATATYPE_UINT_32, qcir::DataType::UINT32},
      {QNN_DATATYPE_UINT_64, qcir::DataType::UINT64},
      {QNN_DATATYPE_FLOAT_16, qcir::DataType::FLOAT16},
      {QNN_DATATYPE_FLOAT_32, qcir::DataType::FLOAT32},
      // {QNN_DATATYPE_FLOAT_64, qcir::DataType::FLOAT64},
      {QNN_DATATYPE_SFIXED_POINT_4, qcir::DataType::SFIXED4},
      {QNN_DATATYPE_SFIXED_POINT_8, qcir::DataType::SFIXED8},
      {QNN_DATATYPE_SFIXED_POINT_16, qcir::DataType::SFIXED16},
      {QNN_DATATYPE_SFIXED_POINT_32, qcir::DataType::SFIXED32},
      {QNN_DATATYPE_UFIXED_POINT_4, qcir::DataType::UFIXED4},
      {QNN_DATATYPE_UFIXED_POINT_8, qcir::DataType::UFIXED8},
      {QNN_DATATYPE_UFIXED_POINT_16, qcir::DataType::UFIXED16},
      {QNN_DATATYPE_UFIXED_POINT_32, qcir::DataType::UFIXED32},
      {QNN_DATATYPE_BOOL_8, qcir::DataType::BOOL},
      // {QNN_DATATYPE_STRING, qcir::DataType::STRING},
      {QNN_DATATYPE_UNDEFINED, qcir::DataType::UNDEFINED},
  };
  return type_map.at(type);
}

// TODO: enable commented type by QNN version control
Qnn_DataType_t ToDataType(qcir::DataType type) {
  static const std::unordered_map<qcir::DataType, Qnn_DataType_t> type_map{
      {qcir::DataType::INT8, QNN_DATATYPE_INT_8},
      {qcir::DataType::INT16, QNN_DATATYPE_INT_16},
      {qcir::DataType::INT32, QNN_DATATYPE_INT_32},
      {qcir::DataType::INT64, QNN_DATATYPE_INT_64},
      {qcir::DataType::UINT8, QNN_DATATYPE_UINT_8},
      {qcir::DataType::UINT16, QNN_DATATYPE_UINT_16},
      {qcir::DataType::UINT32, QNN_DATATYPE_UINT_32},
      {qcir::DataType::UINT64, QNN_DATATYPE_UINT_64},
      {qcir::DataType::FLOAT16, QNN_DATATYPE_FLOAT_16},
      {qcir::DataType::FLOAT32, QNN_DATATYPE_FLOAT_32},
      // {qcir::DataType::FLOAT64, QNN_DATATYPE_FLOAT_64},
      {qcir::DataType::SFIXED4, QNN_DATATYPE_SFIXED_POINT_4},
      {qcir::DataType::SFIXED8, QNN_DATATYPE_SFIXED_POINT_8},
      {qcir::DataType::SFIXED16, QNN_DATATYPE_SFIXED_POINT_16},
      {qcir::DataType::SFIXED32, QNN_DATATYPE_SFIXED_POINT_32},
      {qcir::DataType::UFIXED4, QNN_DATATYPE_UFIXED_POINT_4},
      {qcir::DataType::UFIXED8, QNN_DATATYPE_UFIXED_POINT_8},
      {qcir::DataType::UFIXED16, QNN_DATATYPE_UFIXED_POINT_16},
      {qcir::DataType::UFIXED32, QNN_DATATYPE_UFIXED_POINT_32},
      {qcir::DataType::BOOL, QNN_DATATYPE_BOOL_8},
      // {qcir::DataType::STRING, QNN_DATATYPE_STRING},
      {qcir::DataType::UNDEFINED, QNN_DATATYPE_UNDEFINED},
  };
  return type_map.at(type);
}

flatbuffers::Offset<qcir::QuantizeParam> ToQuantizeParam(
    const Qnn_Tensor_t& tensor,
    flatbuffers::FlatBufferBuilder* builder) {
  static const std::unordered_map<Qnn_Definition_t, qcir::QuantizeDef> def_map{
      {QNN_DEFINITION_IMPL_GENERATED, qcir::QuantizeDef::IMPL_GENERATED},
      {QNN_DEFINITION_DEFINED, qcir::QuantizeDef::DEFINED},
      {QNN_DEFINITION_UNDEFINED, qcir::QuantizeDef::UNDEFINED},
  };
  static const std::
      unordered_map<Qnn_QuantizationEncoding_t, qcir::QuantizeType>
          type_map{
              {QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
               qcir::QuantizeType::SCALE_OFFSET},
              {QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET,
               qcir::QuantizeType::AXIS_SCALE_OFFSET},
              {QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET,
               qcir::QuantizeType::BW_SCALE_OFFSET},
              {QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET,
               qcir::QuantizeType::BW_AXIS_SCALE_OFFSET},
              {QNN_QUANTIZATION_ENCODING_UNDEFINED,
               qcir::QuantizeType::UNDEFINED},
          };

  int32_t axis = 0;
  uint32_t bitwidth = 0;
  auto param = QNN_VER_PTR(tensor)->quantizeParams;
  auto quant_type = type_map.at(param.quantizationEncoding);
  std::vector<qcir::ScaleOffset> data;
  std::vector<float> scales;
  std::vector<int32_t> offsets;
  switch (quant_type) {
    case qcir::QuantizeType::SCALE_OFFSET: {
      data.emplace_back(qcir::ScaleOffset(
          param.scaleOffsetEncoding.scale, param.scaleOffsetEncoding.offset));
    } break;
    case qcir::QuantizeType::AXIS_SCALE_OFFSET: {
      size_t len = param.axisScaleOffsetEncoding.numScaleOffsets;
      axis = param.axisScaleOffsetEncoding.axis;
      data.reserve(len);
      for (uint i = 0; i < len; ++i) {
        data.emplace_back(qcir::ScaleOffset(
            param.axisScaleOffsetEncoding.scaleOffset[i].scale,
            param.axisScaleOffsetEncoding.scaleOffset[i].offset));
      }
    } break;
    case qcir::QuantizeType::BW_SCALE_OFFSET: {
      bitwidth = param.bwScaleOffsetEncoding.bitwidth;
      scales.push_back(param.bwScaleOffsetEncoding.scale);
      offsets.push_back(param.bwScaleOffsetEncoding.offset);
    } break;
    case qcir::QuantizeType::BW_AXIS_SCALE_OFFSET: {
      bitwidth = param.bwAxisScaleOffsetEncoding.bitwidth;
      axis = param.bwAxisScaleOffsetEncoding.axis;
      size_t len = param.bwAxisScaleOffsetEncoding.numElements;
      scales.reserve(len);
      offsets.reserve(len);
      for (size_t i = 0; i < len; ++i) {
        scales.push_back(param.bwAxisScaleOffsetEncoding.scales[i]);
        offsets.push_back(param.bwAxisScaleOffsetEncoding.offsets[i]);
      }
    } break;
    default:
      QNN_EXECUTORCH_LOG_WARN(
          "QNN_QUANTIZATION_ENCODING_UNDEFINED detected: %s",
          QNN_VER_PTR(tensor)->name);
      break;
  }
  return CreateQuantizeParamDirect(
      *builder,
      def_map.at(param.encodingDefinition),
      quant_type,
      bitwidth,
      axis,
      &scales,
      &offsets,
      &data);
}

Qnn_QuantizeParams_t ToQuantizeParam(const tensor_type& tensor) {
  static const std::unordered_map<qcir::QuantizeDef, Qnn_Definition_t> def_map{
      {qcir::QuantizeDef::IMPL_GENERATED, QNN_DEFINITION_IMPL_GENERATED},
      {qcir::QuantizeDef::DEFINED, QNN_DEFINITION_DEFINED},
      {qcir::QuantizeDef::UNDEFINED, QNN_DEFINITION_UNDEFINED},
  };
  static const std::
      unordered_map<qcir::QuantizeType, Qnn_QuantizationEncoding_t>
          type_map{
              {qcir::QuantizeType::SCALE_OFFSET,
               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET},
              {qcir::QuantizeType::AXIS_SCALE_OFFSET,
               QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET},
              {qcir::QuantizeType::BW_SCALE_OFFSET,
               QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET},
              {qcir::QuantizeType::BW_AXIS_SCALE_OFFSET,
               QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET},
              {qcir::QuantizeType::UNDEFINED,
               QNN_QUANTIZATION_ENCODING_UNDEFINED},
          };

  Qnn_QuantizeParams_t p = QNN_QUANTIZE_PARAMS_INIT;
  auto param = tensor->qparam();
  p.encodingDefinition = def_map.at(param->def());
  p.quantizationEncoding = type_map.at(param->type());
  switch (p.quantizationEncoding) {
    case QNN_QUANTIZATION_ENCODING_SCALE_OFFSET: {
      p.scaleOffsetEncoding.scale = param->data()->Get(0)->scale();
      p.scaleOffsetEncoding.offset = param->data()->Get(0)->offset();
    } break;
    case QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET: {
      p.axisScaleOffsetEncoding.axis = param->axis();
      p.axisScaleOffsetEncoding.numScaleOffsets = param->data()->size();
      p.axisScaleOffsetEncoding.scaleOffset =
          reinterpret_cast<Qnn_ScaleOffset_t*>(
              const_cast<uint8_t*>(param->data()->Data()));
    } break;
    case QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET: {
      p.bwAxisScaleOffsetEncoding.bitwidth = param->bitwidth();
      p.bwScaleOffsetEncoding.scale = param->scales()->Get(0);
      p.bwScaleOffsetEncoding.offset = param->offsets()->Get(0);
    } break;
    case QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET: {
      p.bwAxisScaleOffsetEncoding.bitwidth = param->bitwidth();
      p.bwAxisScaleOffsetEncoding.axis = param->axis();
      p.bwAxisScaleOffsetEncoding.numElements = param->scales()->size();
      p.bwAxisScaleOffsetEncoding.scales =
          const_cast<float*>(param->scales()->data());
      p.bwAxisScaleOffsetEncoding.offsets =
          const_cast<int32_t*>(param->offsets()->data());
    } break;
    default:
      QNN_EXECUTORCH_LOG_WARN(
          "qcir::QuantizeType::UNDEFINED detected: %s",
          tensor->name()->c_str());
      break;
  }
  return p;
}

flatbuffers::Offset<qcir::Tensor> ToTensor(
    const Qnn_Tensor_t& tensor,
    flatbuffers::FlatBufferBuilder* builder) {
  std::vector<uint8_t> buffer(
      static_cast<uint8_t*>(QNN_VER_PTR(tensor)->clientBuf.data),
      static_cast<uint8_t*>(QNN_VER_PTR(tensor)->clientBuf.data) +
          QNN_VER_PTR(tensor)->clientBuf.dataSize);
  std::vector<uint32_t> shape(
      QNN_VER_PTR(tensor)->dimensions,
      QNN_VER_PTR(tensor)->dimensions + QNN_VER_PTR(tensor)->rank);

  return qcir::CreateTensorDirect(
      *builder,
      QNN_VER_PTR(tensor)->name,
      &shape,
      ToTensorType(QNN_VER_PTR(tensor)->type),
      ToDataType(QNN_VER_PTR(tensor)->dataType),
      ToQuantizeParam(tensor, builder),
      &buffer);
}

Qnn_Tensor_t ToTensor(const tensor_type& tensor) {
  auto is_io_tensor = [](Qnn_TensorType_t type) {
    return type < QNN_TENSOR_TYPE_STATIC;
  };

  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  QNN_VER_PTR(t)->name = tensor->name()->c_str();
  QNN_VER_PTR(t)->type = ToTensorType(tensor->type());
  QNN_VER_PTR(t)->dataType = ToDataType(tensor->dtype());
  QNN_VER_PTR(t)->quantizeParams = ToQuantizeParam(tensor);
  QNN_VER_PTR(t)->rank = tensor->shape()->size();
  QNN_VER_PTR(t)->dimensions = const_cast<uint32_t*>(tensor->shape()->data());
  QNN_VER_PTR(t)->clientBuf.dataSize = tensor->data()->size();
  QNN_VER_PTR(t)->clientBuf.data = is_io_tensor(QNN_VER_PTR(t)->type)
      ? nullptr
      : static_cast<void*>(const_cast<uint8_t*>(tensor->data()->Data()));
  return t;
}

} // namespace qnn
} // namespace executor
} // namespace torch
