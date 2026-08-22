/*
 *  Copyright (c) 2025 Samsung Electronics Co. LTD
 *  All rights reserved
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree.
 *
 */
#pragma once

#include <stdint.h>
#include <memory>
#include <string>
#include <vector>

#include <include/common-types.h>
namespace torch {
namespace executor {
namespace enn {

class EnnOpWrapper {
 public:
  EnnOpWrapper(
      std::string op_name,
      std::string op_type,
      const std::vector<TENSOR_ID_T>& input_tensors_id,
      const std::vector<TENSOR_ID_T>& output_tensors_id)
      : name_(std::move(op_name)),
        op_type_(std::move(op_type)),
        input_tensors_(input_tensors_id),
        output_tensors_(output_tensors_id) {}

  void AddOpParam(std::shared_ptr<OpParamWrapper> param) {
    params_.emplace_back(std::move(param));
  }

  const std::string& GetName() const {
    return name_;
  }

  const std::string GetType() const {
    return op_type_;
  }

  const std::vector<TENSOR_ID_T>& GetInputs() const {
    return input_tensors_;
  }

  const std::vector<TENSOR_ID_T>& GetOutputs() const {
    return output_tensors_;
  }

  const std::vector<std::shared_ptr<OpParamWrapper>>& GetParams() const {
    return params_;
  }

 private:
  std::string name_;
  std::string op_type_;
  std::vector<TENSOR_ID_T> input_tensors_;
  std::vector<TENSOR_ID_T> output_tensors_;
  std::vector<std::shared_ptr<OpParamWrapper>> params_;
};

} // namespace enn
} // namespace executor
} // namespace torch
