/*
 *  Copyright (c) 2025 Samsung Electronics Co. LTD
 *  All rights reserved
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree.
 *
 */
#pragma once

#include <pybind11/numpy.h>
#include <stdint.h>

#include "quantize_param_wrapper.h"

namespace py = pybind11;

namespace torch {
namespace executor {
namespace enn {

class EnnTensorWrapper {
 public:
  EnnTensorWrapper(
      std::string tensor_name,
      const std::vector<DIM_T>& dims,
      std::string data_type,
      std::string layout)
      : name_(std::move(tensor_name)),
        shape_(dims),
        data_type_(data_type),
        layout_(std::move(layout)) {}

  void AddQuantizeParam(
      std::string quantize_dtype,
      const std::vector<float>& scales,
      const std::vector<int32_t>& zero_points) {
    quantize_param_ = std::make_unique<EnnQuantizeParamWrapper>(
        quantize_dtype, scales, zero_points);
  }

  void AddData(py::array& data) {
    data_bytes_ = data.nbytes();

    if (data.data() == nullptr || data_bytes_ == 0) {
      return;
    }
    data_ = std::unique_ptr<uint8_t[]>(new uint8_t[data_bytes_]);
    memcpy(data_.get(), data.data(), data_bytes_);
  }

  const std::string& GetName() const {
    return name_;
  }
  const std::vector<DIM_T>& GetShape() const {
    return shape_;
  }
  const std::string& GetDataType() const {
    return data_type_;
  }
  const std::string& GetLayout() const {
    return layout_;
  }
  const EnnQuantizeParamWrapper* GetQuantizeParam() const {
    return quantize_param_.get();
  }

  bool HasConstantData() const {
    return data_ != nullptr && data_bytes_ != 0;
  }

  const uint8_t* GetDataRawPtr() const {
    return data_.get();
  }

  uint32_t GetDataBytes() const {
    return data_bytes_;
  }

 private:
  std::string name_;
  std::vector<DIM_T> shape_;
  std::string data_type_;
  std::string layout_;
  std::unique_ptr<EnnQuantizeParamWrapper> quantize_param_ = nullptr;
  std::unique_ptr<uint8_t[]> data_ = nullptr;
  uint32_t data_bytes_ = 0;
};

} // namespace enn
} // namespace executor
} // namespace torch
