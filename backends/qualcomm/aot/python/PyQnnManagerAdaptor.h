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
#include <executorch/backends/qualcomm/qc_binary_info_generated.h>
#include <executorch/backends/qualcomm/qc_compiler_spec_generated.h>
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/backends/qualcomm/runtime/QnnManager.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <string_view>

namespace py = pybind11;
namespace executorch {
namespace backends {
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
    qnn_executorch_context_binary_.buffer = info.ptr;
    qnn_executorch_context_binary_.nbytes = info.size * info.itemsize;
    qnn_manager_ = std::make_shared<QnnManager>(
        qnn_executorch_options, qnn_executorch_context_binary_);
  }

  // used for loading multiple graphs in qcir
  explicit PyQnnManager(const py::bytes& buffer, const py::list& qcirs)
      : qnn_executorch_option_ptr_(buffer) {
    auto qnn_executorch_options = GetQnnExecuTorchOptions(
        qnn_executorch_option_ptr_.cast<std::string_view>().data());

    // merge multiple qcirs into one context with multiple graphs
    std::vector<flatbuffers::Offset<qcir::Graph>> graphs;
    for (size_t i = 0; i < qcirs.size(); ++i) {
      py::buffer_info info(py::buffer(qcirs[i].cast<py::bytes>()).request());
      flatbuffers::Verifier verifier_binary_info(
          static_cast<const uint8_t* const>(info.ptr),
          info.size * info.itemsize);
      if (!qnn_delegate::VerifyBinaryInfoBuffer(verifier_binary_info)) {
        QNN_EXECUTORCH_LOG_ERROR("Fail to verify binary info");
        return;
      }
      auto binary_info = qnn_delegate::GetBinaryInfo(info.ptr);

      flatbuffers::Verifier verifier_qcir(
          binary_info->data()->data(), binary_info->data()->size());
      if (!qcir::VerifyContextBuffer(verifier_qcir)) {
        QNN_EXECUTORCH_LOG_ERROR("Fail to verify qcir format");
        return;
      }
      auto context = qcir::GetContext(binary_info->data()->data());
      for (const auto& graph : *context->graphs()) {
        std::vector<flatbuffers::Offset<qcir::Tensor>> tensors;
        for (const auto tensor : *graph->tensors()) {
          // here we need to take a detour to merge multiple qcir flatbuffers
          // outer ToTensor
          //   return: flatbuffers::Offset<Tensor>
          //   consume: QnnTensor, flatbuffers::FlatBufferBuilder*
          // inner ToTensor
          //   return: QnnTensor
          //   consume: flatbuffers::Vector<::flatbuffers::Offset<qcir::Tensor>>
          tensors.emplace_back(ToTensor(ToTensor(tensor), &builder_));
        }
        std::vector<flatbuffers::Offset<qcir::Operator>> nodes;
        for (const auto& node : *graph->nodes()) {
          int32_t* inputs_ptr = const_cast<int32_t*>(node->inputs()->data());
          int32_t* outputs_ptr = const_cast<int32_t*>(node->outputs()->data());
          int32_t* params_ptr = const_cast<int32_t*>(node->params()->data());
          std::vector<int32_t> inputs(
              inputs_ptr, inputs_ptr + node->inputs()->size());
          std::vector<int32_t> outputs(
              outputs_ptr, outputs_ptr + node->outputs()->size());
          std::vector<int32_t> params(
              params_ptr, params_ptr + node->params()->size());
          nodes.emplace_back(qcir::CreateOperatorDirect(
              builder_,
              node->name()->str().c_str(),
              node->package_name()->str().c_str(),
              node->type_name()->str().c_str(),
              &inputs,
              &outputs,
              &params));
        }
        graphs.emplace_back(qcir::CreateGraphDirect(
            builder_, graph->name()->str().c_str(), &nodes, &tensors));
      }
    }

    auto context = qcir::CreateContextDirect(builder_, &graphs);
    builder_.Finish(context);
    QnnExecuTorchContextBinary qcir_bin(
        {builder_.GetBufferPointer(), builder_.GetSize()});

    qnn_executorch_context_binary_ = MakeBinaryInfo(qcir_bin);
    qnn_manager_ = std::make_shared<QnnManager>(
        qnn_executorch_options, qnn_executorch_context_binary_);
  }

  executorch::runtime::Error Init() {
    return qnn_manager_->Init();
  }

  bool IsNodeSupportedByBackend(
      std::vector<std::shared_ptr<OpWrapper>>& op_wrappers) {
    return qnn_manager_->IsNodeSupportedByBackend(op_wrappers);
  }

  // this method is specific for compiling multi-graphs
  py::array_t<char> Compile() {
    if (qnn_manager_->CompileQcir() != Error::Ok) {
      QNN_EXECUTORCH_LOG_ERROR("Fail to compile qcir");
      return py::array_t<char>(0);
    }

    // generate context binary if compilation succeded
    QnnExecuTorchContextBinary binary_info;
    qnn_manager_->GetContextBinary(binary_info);
    // allocate py::array (to pass the result of the C++ function to Python)
    auto result = py::array_t<char>(binary_info.nbytes);
    auto result_buffer = result.request();
    char* result_ptr = (char*)result_buffer.ptr;
    std::memcpy(result_ptr, binary_info.buffer, binary_info.nbytes);
    return result;
  }

  py::array_t<char> Compile(
      const std::string& graph_name,
      std::vector<std::shared_ptr<OpWrapper>>& op_wrappers) {
    QnnExecuTorchContextBinary binary_info;

    if (qnn_manager_->IsOnlinePrepare() || qnn_manager_->IsMultipleGraphs()) {
      builder_.Reset();
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
              ToTensor(wrapper->CloneTensorStruct(), &builder_));
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
            executorch::runtime::Error err = param->PopulateQnnParam();
            if (err != executorch::runtime::Error::Ok) {
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
            tensors.emplace_back(ToTensor(t, &builder_));
          }
        }

        Qnn_OpConfig_t op_config = op_wrapper->GetOpConfig();
        operators.emplace_back(qcir::CreateOperatorDirect(
            builder_,
            QNN_VER_PTR(op_config)->name,
            QNN_VER_PTR(op_config)->packageName,
            QNN_VER_PTR(op_config)->typeName,
            &inputs,
            &outputs,
            &params));
      }
      auto graph = qcir::CreateGraphDirect(
          builder_, graph_name.c_str(), &operators, &tensors);
      std::vector<flatbuffers::Offset<qcir::Graph>> graphs({graph});
      auto context = qcir::CreateContextDirect(builder_, &graphs);
      builder_.Finish(context);
      QnnExecuTorchContextBinary qcir_binary(
          {builder_.GetBufferPointer(), builder_.GetSize()});
      binary_info = MakeBinaryInfo(qcir_binary);
    } else {
      if (qnn_manager_->Compile(graph_name, op_wrappers) !=
          executorch::runtime::Error::Ok) {
        QNN_EXECUTORCH_LOG_ERROR("Fail to compile QNN graph");
        return py::array_t<char>(0);
      }
      if (qnn_manager_->GetContextBinary(binary_info) !=
          executorch::runtime::Error::Ok) {
        return py::array_t<char>(0);
      }
    }

    // allocate py::array (to pass the result of the C++ function to Python)
    auto result = py::array_t<char>(binary_info.nbytes);
    auto result_buffer = result.request();
    char* result_ptr = (char*)result_buffer.ptr;
    std::memcpy(result_ptr, binary_info.buffer, binary_info.nbytes);
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

  executorch::runtime::Error AllocateTensor(const std::string& graph_name) {
    return qnn_manager_->AllocateTensor(graph_name);
  }

  py::list GetGraphInputs(const std::string& graph_name) {
    py::list ret;
    for (const std::shared_ptr<TensorWrapper>& input :
         qnn_manager_->GetGraphInputs(graph_name)) {
      ret.append(PyQnnTensorWrapper(input));
    }
    return ret;
  }

  py::list GetGraphOutputs(const std::string& graph_name) {
    py::list ret;
    for (const std::shared_ptr<TensorWrapper>& output :
         qnn_manager_->GetGraphOutputs(graph_name)) {
      ret.append(PyQnnTensorWrapper(output));
    }
    return ret;
  }

  py::list GetGraphNames() {
    py::list ret;
    for (const std::string& graph_name : qnn_manager_->GetGraphNames()) {
      ret.append(graph_name);
    }
    return ret;
  }

  uint64_t GetSpillFillBufferSize() {
    return qnn_manager_->GetSpillFillBufferSize();
  }

  py::array_t<char> MakeBinaryInfo(const py::bytes& ctx_bin) {
    py::buffer_info info(py::buffer(ctx_bin).request());
    QnnExecuTorchContextBinary binary(
        {info.ptr, static_cast<uint64_t>(info.size * info.itemsize)});
    auto binary_info = MakeBinaryInfo(binary);
    auto result = py::array_t<char>(binary_info.nbytes);
    auto result_buffer = result.request();
    std::memcpy(result_buffer.ptr, binary_info.buffer, binary_info.nbytes);
    return result;
  }

 private:
  QnnExecuTorchContextBinary MakeBinaryInfo(
      const QnnExecuTorchContextBinary& ctx_bin) {
    auto signature = []() {
      return std::to_string(
          std::chrono::high_resolution_clock::now().time_since_epoch().count());
    };
    const uint8_t* base = static_cast<uint8_t*>(ctx_bin.buffer);
    std::vector<uint8_t> data(base, base + ctx_bin.nbytes);
    // add signature to binary for cache reuse in runtime
    builder_.Reset();
    auto binary_info = qnn_delegate::CreateBinaryInfoDirect(
        builder_, signature().c_str(), &data);
    builder_.Finish(binary_info);

    return QnnExecuTorchContextBinary(
        {builder_.GetBufferPointer(), builder_.GetSize()});
  }

  // Store the bytes object instead of a raw pointer so that this module will
  // keep the bytes alive.
  const py::bytes qnn_executorch_option_ptr_;
  QnnExecuTorchContextBinary qnn_executorch_context_binary_;
  std::shared_ptr<QnnManager> qnn_manager_;
  flatbuffers::FlatBufferBuilder builder_;
};
} // namespace qnn
} // namespace backends
} // namespace executorch
