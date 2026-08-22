/*
 *  Copyright (c) 2025 Samsung Electronics Co. LTD
 *  All rights reserved
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree.
 *
 */
#pragma once

#include <include/common-types.h>
#include <include/graph_wrapper_api.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <exception>
#include <iostream>

#include "wrappers/op_param_wrapper.h"
#include "wrappers/op_wrapper.h"
#include "wrappers/tensor_wrapper.h"

namespace py = pybind11;

namespace torch {
namespace executor {
namespace enn {

class PyEnnGraphWrapper {
 public:
  PyEnnGraphWrapper() {}

  void Init() {
    graph_wrapper_ = create_graph("");
  }

  TENSOR_ID_T DefineTensor(std::shared_ptr<EnnTensorWrapper> tensor) const {
    TENSOR_ID_T tensor_id;
    auto result = define_tensor(
        graph_wrapper_,
        &tensor_id,
        tensor->GetName().c_str(),
        tensor->GetShape().data(),
        tensor->GetShape().size(),
        tensor->GetDataType().c_str(),
        tensor->GetLayout().c_str());
    if (result != GraphWrapperReturn::SUCCESS) {
      throw std::runtime_error("fail in define tensor");
    }

    if (tensor->HasConstantData()) {
      auto set_data_result = set_data_for_constant_tensor(
          graph_wrapper_,
          tensor_id,
          tensor->GetDataRawPtr(),
          tensor->GetDataBytes());
      if (set_data_result != GraphWrapperReturn::SUCCESS) {
        throw std::runtime_error("fail in define tensor");
      }
    }

    auto* quantize_param = tensor->GetQuantizeParam();
    if (quantize_param != nullptr) {
      auto set_qparam_result = set_quantize_param_for_tensor(
          graph_wrapper_,
          tensor_id,
          quantize_param->GetQuantizeType().c_str(),
          quantize_param->GetScales(),
          quantize_param->GetZeroPoints());
      if (set_qparam_result != GraphWrapperReturn::SUCCESS) {
        throw std::runtime_error("fail in define tensor");
      }
    }

    return tensor_id;
  }

  NODE_ID_T DefineOpNode(std::shared_ptr<EnnOpWrapper> op) const {
    NODE_ID_T op_id;

    auto result = define_op_node(
        graph_wrapper_,
        &op_id,
        op->GetName().c_str(),
        op->GetType().c_str(),
        op->GetInputs().data(),
        op->GetInputs().size(),
        op->GetOutputs().data(),
        op->GetOutputs().size());
    if (result != GraphWrapperReturn::SUCCESS) {
      throw std::runtime_error("fail in define op");
    }

    for (const auto& param : op->GetParams()) {
      add_op_parameter(
          graph_wrapper_, op_id, param->getKeyName().c_str(), param->Dump());
    }

    return op_id;
  }

  void SetGraphInputTensors(const std::vector<TENSOR_ID_T>& tensors) const {
    auto result =
        set_graph_input_tensors(graph_wrapper_, tensors.data(), tensors.size());
    if (result != GraphWrapperReturn::SUCCESS) {
      throw std::runtime_error("fail in set graph inputs");
    }
  }

  void SetGraphOutputTensors(const std::vector<TENSOR_ID_T>& tensors) const {
    auto result = set_graph_output_tensors(
        graph_wrapper_, tensors.data(), tensors.size());
    if (result != GraphWrapperReturn::SUCCESS) {
      throw std::runtime_error("fail in set graph outputs");
    }
  }

  void FinishBuild() const {
    auto result = finish_build_graph(graph_wrapper_);

    if (result != GraphWrapperReturn::SUCCESS) {
      throw std::runtime_error("fail to build graph");
    }
  }

  py::array_t<char> Serialize() {
    uint64_t nbytes = 0;
    uint8_t* addr = nullptr;
    auto result = serialize(graph_wrapper_, &addr, &nbytes);

    if (result != GraphWrapperReturn::SUCCESS || addr == nullptr ||
        nbytes == 0) {
      throw std::runtime_error("fail to serialize");
    }

    auto serial_buf = py::array_t<char>(nbytes);
    auto serial_buf_block = serial_buf.request();
    char* serial_buf_ptr = (char*)serial_buf_block.ptr;
    std::memcpy(serial_buf_ptr, addr, nbytes);

    return serial_buf;
  }

  ~PyEnnGraphWrapper() {
    release_graph(graph_wrapper_);
  }

 private:
  GraphHandler graph_wrapper_;
};

} // namespace enn
} // namespace executor
} // namespace torch
