/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <executorch/backends/qualcomm/aot/ir/qcir_utils.h>
#include <executorch/backends/qualcomm/aot/python/PyQnnWrapperAdaptor.h>
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/backends/qualcomm/runtime/QnnManager.h>
#include <executorch/backends/qualcomm/schema_generated.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <string_view>

namespace py = pybind11;
namespace torch {
namespace executor {
namespace qnn {
class PyQnnManager {
 public:
  // used for AoT compilation
  explicit PyQnnManager(const py::bytes& buffer)
      : qnn_executorch_option_ptr_(buffer),
        qnn_executorch_context_binary_(QNN_EXECUTORCH_CONTEXT_BINARY) {
    // Choose non-allocating non-owning string pieces exposed as string_view for
    // parsers
    auto qnn_executorch_options = GetQnnExecuTorchOptions(
        qnn_executorch_option_ptr_.cast<std::string_view>().data());
    qnn_manager_ = std::make_shared<QnnManager>(
        qnn_executorch_options, qnn_executorch_context_binary_);
  }
  // used for loading context binary directly
  explicit PyQnnManager(const py::bytes& buffer, const py::bytes& ctx_bin)
      : qnn_executorch_option_ptr_(buffer) {
    auto qnn_executorch_options = GetQnnExecuTorchOptions(
        qnn_executorch_option_ptr_.cast<std::string_view>().data());

    py::buffer_info info(py::buffer(ctx_bin).request());
    qnn_executorch_context_binary_.buffer = static_cast<void*>(info.ptr);
    qnn_executorch_context_binary_.nbytes = info.size * info.itemsize;
    qnn_manager_ = std::make_shared<QnnManager>(
        qnn_executorch_options, qnn_executorch_context_binary_);
  }

  Error Init() {
    return qnn_manager_->Init();
  }
  bool IsNodeSupportedByBackend(
      std::vector<std::shared_ptr<OpWrapper>>& op_wrappers) {
    return qnn_manager_->IsNodeSupportedByBackend(op_wrappers);
  }
  py::array_t<char> Compile(
      std::vector<std::shared_ptr<OpWrapper>>& op_wrappers) {
    QnnExecuTorchContextBinary context_binary;
    flatbuffers::FlatBufferBuilder builder;

    if (qnn_manager_->IsOnlinePrepare()) {
      std::vector<flatbuffers::Offset<qcir::Tensor>> tensors;
      std::unordered_map<void*, int> tensor_map;

      auto set_tensor = [&](const std::shared_ptr<TensorWrapper>& wrapper,
                            std::vector<int>& index) {
        auto it = tensor_map.find(wrapper.get());
        if (it != tensor_map.end()) {
          index.push_back(it->second);
        } else {
          int i = tensors.size();
          tensor_map[wrapper.get()] = i;
          index.push_back(i);
          tensors.emplace_back(
              ToTensor(wrapper->CloneTensorStruct(), &builder));
        }
      };

      std::vector<flatbuffers::Offset<qcir::Operator>> operators;
      for (std::shared_ptr<OpWrapper>& op_wrapper : op_wrappers) {
        std::vector<int> inputs, outputs, params;

        for (const auto& tensor_wrapper : op_wrapper->GetInputTensors()) {
          set_tensor(tensor_wrapper, inputs);
        }

        for (const auto& tensor_wrapper : op_wrapper->GetOutputTensors()) {
          set_tensor(tensor_wrapper, outputs);
        }

        for (const auto& param : op_wrapper->GetParams()) {
          auto* p_tensor_param = dynamic_cast<TensorParamWrapper*>(param.get());
          if (p_tensor_param != nullptr) {
            auto wrapper = p_tensor_param->GetTensorWrapper();
            wrapper->SetName(param->GetName());
            set_tensor(wrapper, params);
          } else {
            Error err = param->PopulateQnnParam();
            if (err != Error::Ok) {
              QNN_EXECUTORCH_LOG_ERROR(
                  "Fail to get scalar parameter in online prepare stage");
              return py::array_t<char>(0);
            }
            Qnn_Param_t p = param->GetQnnParam();
            Qnn_Tensor_t t = QNN_TENSOR_INIT;
            QNN_VER_PTR(t)->name = p.name;
            QNN_VER_PTR(t)->dataType = p.scalarParam.dataType;
            QNN_VER_PTR(t)->clientBuf.data =
                static_cast<void*>(&p.scalarParam.uint8Value);
            QNN_VER_PTR(t)->clientBuf.dataSize =
                GetDataTypeSize(QNN_VER_PTR(t)->dataType);
            params.push_back(tensors.size());
            tensors.emplace_back(ToTensor(t, &builder));
          }
        }

        Qnn_OpConfig_t op_config = op_wrapper->GetOpConfig();
        operators.emplace_back(qcir::CreateOperatorDirect(
            builder,
            QNN_VER_PTR(op_config)->name,
            QNN_VER_PTR(op_config)->packageName,
            QNN_VER_PTR(op_config)->typeName,
            &inputs,
            &outputs,
            &params));
      }
      auto graph = qcir::CreateGraphDirect(builder, &operators, &tensors);
      builder.Finish(graph);
      context_binary.buffer = builder.GetBufferPointer();
      context_binary.nbytes = builder.GetSize();
    } else if (
        qnn_manager_->Compile(op_wrappers, context_binary) != Error::Ok) {
      return py::array_t<char>(0);
    }

    // allocate py::array (to pass the result of the C++ function to
    // Python)
    auto result = py::array_t<char>(context_binary.nbytes);
    auto result_buffer = result.request();
    char* result_ptr = (char*)result_buffer.ptr;
    std::memcpy(result_ptr, context_binary.buffer, context_binary.nbytes);
    return result;
  }
  void Destroy() {
    return qnn_manager_->Destroy();
  }

  bool IsAvailable() {
    return qnn_manager_->IsAvailable();
  }

  bool IsTensorDump() {
    return qnn_manager_->IsTensorDump();
  }

  Error AllocateTensor() {
    return qnn_manager_->AllocateTensor();
  }

  py::list GetGraphInputs() {
    py::list ret;
    for (const std::shared_ptr<TensorWrapper>& input :
         qnn_manager_->GetGraphInputs()) {
      ret.append(PyQnnTensorWrapper(input));
    }
    return ret;
  }

  py::list GetGraphOutputs() {
    py::list ret;
    for (const std::shared_ptr<TensorWrapper>& output :
         qnn_manager_->GetGraphOutputs()) {
      ret.append(PyQnnTensorWrapper(output));
    }
    return ret;
  }

  uint64_t GetSpillFillBufferSize() {
    return qnn_manager_->GetSpillFillBufferSize();
  }

 private:
  // Store the bytes object instead of a raw pointer so that this module will
  // keep the bytes alive.
  const py::bytes qnn_executorch_option_ptr_;
  QnnExecuTorchContextBinary qnn_executorch_context_binary_;
  std::shared_ptr<QnnManager> qnn_manager_;
};
} // namespace qnn
} // namespace executor
} // namespace torch
