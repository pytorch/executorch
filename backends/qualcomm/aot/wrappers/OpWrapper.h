/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/aot/wrappers/ParamWrapper.h>
#include <executorch/backends/qualcomm/aot/wrappers/QuantizeParamsWrapper.h>
#include <executorch/backends/qualcomm/aot/wrappers/ScalarParamWrapper.h>
#include <executorch/backends/qualcomm/aot/wrappers/TensorParamWrapper.h>
#include <executorch/backends/qualcomm/aot/wrappers/TensorWrapper.h>

#include <cstdint>
#include <memory>
#include <sstream>
#include <typeinfo>
namespace torch {
namespace executor {
namespace qnn {
class OpWrapper final {
 public:
  explicit OpWrapper(
      std::string name,
      std::string package_name,
      std::string op_type)
      : name_(std::move(name)),
        package_name_(std::move(package_name)),
        op_type_(std::move(op_type)) {}

  OpWrapper(OpWrapper&& other) noexcept
      : name_(std::move(other.name_)),
        package_name_(std::move(other.package_name_)),
        op_type_(std::move(other.op_type_)),
        input_tensors_(std::move(other.input_tensors_)),
        output_tensors_(std::move(other.output_tensors_)),
        params_(std::move(other.params_)),
        input_tensor_structs_(std::move(other.input_tensor_structs_)),
        output_tensor_structs_(std::move(other.output_tensor_structs_)) {}

  OpWrapper(const OpWrapper& other) = delete;

  OpWrapper& operator=(const OpWrapper& other) = delete;

  OpWrapper& operator=(OpWrapper&& other) = delete;

  ~OpWrapper() = default;

  void AddInputTensors(
      const std::vector<std::shared_ptr<TensorWrapper>>& tensors) {
    input_tensors_ = tensors;
  }

  void AddOutputTensors(
      const std::vector<std::shared_ptr<TensorWrapper>>& tensors) {
    output_tensors_ = tensors;
  }

  void AddTensorParam(
      const std::string& name,
      Qnn_DataType_t data_type,
      std::uint32_t rank,
      const std::uint32_t dims[],
      const void* data,
      bool copy_data = false) {
    std::unique_ptr<QuantizeParamsWrapper> quantize_param_wrapper =
        std::make_unique<UndefinedQuantizeParamsWrapper>();
    constexpr std::uint32_t kBytes = 0;
    std::shared_ptr<TensorWrapper> tensor_wrapper = CreateTensorWrapper(
        QNN_TENSOR_TYPE_STATIC,
        data_type,
        std::move(quantize_param_wrapper),
        rank,
        dims,
        kBytes,
        data,
        copy_data);
    params_.emplace_back(
        std::make_unique<TensorParamWrapper>(name, tensor_wrapper));
  }

  template <typename T>
  void
  AddScalarParam(const std::string& name, Qnn_DataType_t data_type, T data) {
    params_.emplace_back(
        std::make_unique<ScalarParamWrapper<T>>(name, data_type, data));
  }

  const std::vector<std::unique_ptr<ParamWrapper>>& GetParams() {
    return params_;
  }

  const std::vector<std::shared_ptr<TensorWrapper>>& GetInputTensors() {
    return input_tensors_;
  }

  const std::vector<std::shared_ptr<TensorWrapper>>& GetOutputTensors() {
    return output_tensors_;
  }
  const std::string GetOpType() {
    return op_type_;
  }
  Qnn_OpConfig_t GetOpConfig();

 private:
  std::string name_;
  std::string package_name_;
  std::string op_type_;
  std::vector<std::shared_ptr<TensorWrapper>> input_tensors_;
  std::vector<std::shared_ptr<TensorWrapper>> output_tensors_;
  std::vector<std::unique_ptr<ParamWrapper>> params_;
  std::vector<Qnn_Tensor_t> input_tensor_structs_;
  std::vector<Qnn_Tensor_t> output_tensor_structs_;
  std::vector<Qnn_Param_t> param_types_;
};
} // namespace qnn
} // namespace executor
} // namespace torch
