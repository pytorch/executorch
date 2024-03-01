/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/aot/wrappers/ParamWrapper.h>
#include <executorch/runtime/core/error.h>
namespace torch {
namespace executor {
namespace qnn {
template <typename T>
class ScalarParamWrapper final : public ParamWrapper {
 public:
  explicit ScalarParamWrapper(
      std::string name,
      Qnn_DataType_t data_type,
      T data)
      : ParamWrapper(QNN_PARAMTYPE_SCALAR, std::move(name)),
        data_type_(data_type),
        data_(data) {}

  // Populate appropriate field in Qnn scalarParam depending on the datatype
  // of the scalar
  Error PopulateQnnParam() override {
    qnn_param_.scalarParam.dataType = data_type_;
    switch (data_type_) {
      case QNN_DATATYPE_BOOL_8:
        qnn_param_.scalarParam.bool8Value = data_;
        break;
      case QNN_DATATYPE_UINT_8:
        qnn_param_.scalarParam.uint8Value = data_;
        break;
      case QNN_DATATYPE_INT_8:
        qnn_param_.scalarParam.int8Value = data_;
        break;
      case QNN_DATATYPE_UINT_16:
        qnn_param_.scalarParam.uint16Value = data_;
        break;
      case QNN_DATATYPE_INT_16:
        qnn_param_.scalarParam.int16Value = data_;
        break;
      case QNN_DATATYPE_UINT_32:
        qnn_param_.scalarParam.uint32Value = data_;
        break;
      case QNN_DATATYPE_INT_32:
        qnn_param_.scalarParam.int32Value = data_;
        break;
      case QNN_DATATYPE_FLOAT_32:
        qnn_param_.scalarParam.floatValue = data_;
        break;
      default:
        QNN_EXECUTORCH_LOG_ERROR(
            "ScalarParamWrapper failed to assign scalarParam value - "
            "invalid datatype %d",
            data_type_);
        return Error::Internal;
    }
    return Error::Ok;
  }

  const T& GetData() const {
    return data_;
  };

 private:
  Qnn_DataType_t data_type_;
  T data_;
};
} // namespace qnn
} // namespace executor
} // namespace torch
