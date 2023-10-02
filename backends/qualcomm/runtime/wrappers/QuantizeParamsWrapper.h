/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "QnnTypes.h"
namespace torch {
namespace executor {
namespace qnn {
class QuantizeParamsWrapper {
 public:
  // To create the QuantizeParams_t using data from this class:
  virtual Qnn_QuantizeParams_t CreateQuantizeParams() = 0;
  // Other accessors:
  Qnn_Definition_t GetEncodingDefinition() const {
    return encoding_definition_;
  };
  Qnn_QuantizationEncoding_t GetQuantizationEncoding() const {
    return quantization_encoding_;
  };

  virtual std::unique_ptr<QuantizeParamsWrapper> Clone() = 0;
  virtual ~QuantizeParamsWrapper() = default;

  QuantizeParamsWrapper(QuantizeParamsWrapper&& rhs) = default;
  QuantizeParamsWrapper(const QuantizeParamsWrapper& rhs) = default;
  QuantizeParamsWrapper& operator=(const QuantizeParamsWrapper& rhs) = default;
  QuantizeParamsWrapper& operator=(QuantizeParamsWrapper&& rhs) = default;

 protected:
  explicit QuantizeParamsWrapper(
      Qnn_Definition_t encoding_definition,
      Qnn_QuantizationEncoding_t quantization_encoding)
      : encoding_definition_(encoding_definition),
        quantization_encoding_(quantization_encoding) {}

 private:
  Qnn_Definition_t encoding_definition_;
  Qnn_QuantizationEncoding_t quantization_encoding_;
};

class UndefinedQuantizeParamsWrapper final : public QuantizeParamsWrapper {
 public:
  UndefinedQuantizeParamsWrapper()
      : QuantizeParamsWrapper(
            QNN_DEFINITION_UNDEFINED,
            QNN_QUANTIZATION_ENCODING_UNDEFINED) {}
  UndefinedQuantizeParamsWrapper(const UndefinedQuantizeParamsWrapper& rhs)
      : QuantizeParamsWrapper(
            rhs.GetEncodingDefinition(),
            rhs.GetQuantizationEncoding()) {}
  UndefinedQuantizeParamsWrapper(UndefinedQuantizeParamsWrapper&& rhs) = delete;
  UndefinedQuantizeParamsWrapper& operator=(
      const UndefinedQuantizeParamsWrapper& rhs) = delete;
  UndefinedQuantizeParamsWrapper& operator=(
      UndefinedQuantizeParamsWrapper&& rhs) = delete;

  ~UndefinedQuantizeParamsWrapper() override = default;

  std::unique_ptr<QuantizeParamsWrapper> Clone() override {
    return std::make_unique<UndefinedQuantizeParamsWrapper>(*this);
  }

  Qnn_QuantizeParams_t CreateQuantizeParams() override {
    Qnn_QuantizeParams_t rval = {
        .encodingDefinition = GetEncodingDefinition(),
        .quantizationEncoding = GetQuantizationEncoding()};
    return rval;
  }
};

class ScaleOffsetQuantizeParamsWrapper final : public QuantizeParamsWrapper {
 public:
  explicit ScaleOffsetQuantizeParamsWrapper(float scale, std::int32_t offset)
      : QuantizeParamsWrapper(
            QNN_DEFINITION_DEFINED,
            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET),
        scale_(scale),
        offset_(offset) {}

  ScaleOffsetQuantizeParamsWrapper(const ScaleOffsetQuantizeParamsWrapper& rhs)
      : QuantizeParamsWrapper(
            rhs.GetEncodingDefinition(),
            rhs.GetQuantizationEncoding()),
        scale_(rhs.scale_),
        offset_(rhs.offset_) {}
  ScaleOffsetQuantizeParamsWrapper(ScaleOffsetQuantizeParamsWrapper&& rhs) =
      delete;
  ScaleOffsetQuantizeParamsWrapper& operator=(
      const ScaleOffsetQuantizeParamsWrapper& rhs) = delete;
  ScaleOffsetQuantizeParamsWrapper& operator=(
      ScaleOffsetQuantizeParamsWrapper&& rhs) = delete;

  ~ScaleOffsetQuantizeParamsWrapper() override = default;

  std::unique_ptr<QuantizeParamsWrapper> Clone() override {
    return std::make_unique<ScaleOffsetQuantizeParamsWrapper>(*this);
  }

  Qnn_QuantizeParams_t CreateQuantizeParams() override {
    Qnn_QuantizeParams_t rval;
    rval.encodingDefinition = GetEncodingDefinition();
    rval.quantizationEncoding = GetQuantizationEncoding();
    rval.scaleOffsetEncoding.scale = scale_;
    rval.scaleOffsetEncoding.offset = offset_;
    return rval;
  }

 private:
  float scale_;
  std::int32_t offset_;
};

class AxisScaleOffsetQuantizeParamsWrapper final
    : public QuantizeParamsWrapper {
 public:
  explicit AxisScaleOffsetQuantizeParamsWrapper(
      std::int32_t axis,
      const std::vector<Qnn_ScaleOffset_t>& scale_offsets)
      : QuantizeParamsWrapper(
            QNN_DEFINITION_DEFINED,
            QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET),
        axis_(axis),
        scale_offsets_(scale_offsets) {}

  AxisScaleOffsetQuantizeParamsWrapper(
      const AxisScaleOffsetQuantizeParamsWrapper& rhs)
      : QuantizeParamsWrapper(
            rhs.GetEncodingDefinition(),
            rhs.GetQuantizationEncoding()),
        axis_(rhs.axis_),
        scale_offsets_(rhs.scale_offsets_) {}
  AxisScaleOffsetQuantizeParamsWrapper(
      AxisScaleOffsetQuantizeParamsWrapper&& rhs) = delete;
  AxisScaleOffsetQuantizeParamsWrapper& operator=(
      const AxisScaleOffsetQuantizeParamsWrapper& rhs) = delete;
  AxisScaleOffsetQuantizeParamsWrapper& operator=(
      AxisScaleOffsetQuantizeParamsWrapper&& rhs) = delete;

  ~AxisScaleOffsetQuantizeParamsWrapper() override = default;

  void SetAxis(std::int32_t axis) {
    axis_ = axis;
  }

  std::unique_ptr<QuantizeParamsWrapper> Clone() override {
    return std::make_unique<AxisScaleOffsetQuantizeParamsWrapper>(*this);
  }

  Qnn_QuantizeParams_t CreateQuantizeParams() override {
    Qnn_QuantizeParams_t rval;
    rval.encodingDefinition = GetEncodingDefinition();
    rval.quantizationEncoding = GetQuantizationEncoding();
    rval.axisScaleOffsetEncoding.axis = axis_;
    rval.axisScaleOffsetEncoding.numScaleOffsets = scale_offsets_.size();
    rval.axisScaleOffsetEncoding.scaleOffset = scale_offsets_.data();
    return rval;
  }

 private:
  std::int32_t axis_;
  std::vector<Qnn_ScaleOffset_t> scale_offsets_;
};

// Factory function to create quantization param wrapper from QnnQuantization
std::unique_ptr<QuantizeParamsWrapper> CreateQuantizationParamWrapper(
    const Qnn_QuantizeParams_t& quantization);
} // namespace qnn
} // namespace executor
} // namespace torch
