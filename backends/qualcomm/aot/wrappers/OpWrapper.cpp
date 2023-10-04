/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/aot/wrappers/OpWrapper.h>
namespace torch {
namespace executor {
namespace qnn {
Qnn_OpConfig_t OpWrapper::GetOpConfig() {
  param_types_.clear();
  input_tensor_structs_.clear();
  output_tensor_structs_.clear();

  for (const auto& param_wrapper : params_) {
    param_types_.emplace_back(param_wrapper->GetQnnParam());
  }
  for (const auto& tensor_wrapper : input_tensors_) {
    input_tensor_structs_.emplace_back(tensor_wrapper->CloneTensorStruct());
  }

  for (const auto& tensor_wrapper : output_tensors_) {
    output_tensor_structs_.emplace_back(tensor_wrapper->CloneTensorStruct());
  }

  Qnn_OpConfig_t ret = QNN_OPCONFIG_INIT;
  ret.version = QNN_OPCONFIG_VERSION_1;
  Qnn_OpConfigV1_t& op_config = ret.v1;

  op_config.name = name_.c_str();
  op_config.packageName = package_name_.c_str();
  op_config.typeName = op_type_.c_str();
  op_config.numOfParams = static_cast<std::uint32_t>(param_types_.size());
  op_config.params = param_types_.data();
  op_config.numOfInputs =
      static_cast<std::uint32_t>(input_tensor_structs_.size());
  op_config.inputTensors = input_tensor_structs_.data();
  op_config.numOfOutputs =
      static_cast<std::uint32_t>(output_tensor_structs_.size());
  op_config.outputTensors = output_tensor_structs_.data();

  return ret;
}
} // namespace qnn
} // namespace executor
} // namespace torch
