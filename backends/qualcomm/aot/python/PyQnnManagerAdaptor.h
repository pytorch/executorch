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
#include <executorch/backends/qualcomm/qc_compiler_spec_generated.h>
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/backends/qualcomm/runtime/QnnManager.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnCustomProtocol.h>
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

  // used during stage 2 of multi-graph mode
  explicit PyQnnManager(const py::bytes& buffer, const py::list& qcirs)
      : qnn_executorch_option_ptr_(buffer) {
    auto qnn_executorch_options = GetQnnExecuTorchOptions(
        qnn_executorch_option_ptr_.cast<std::string_view>().data());

    // merge multiple qcirs into one context with multiple graphs

    // We start retrieving tensor from offsets = 0.
    std::vector<uint32_t> offsets(1, 0);
    std::vector<uint8_t> tensor_data;
    std::vector<uint8_t*> tensor_ptr;
    std::vector<uint64_t> tensor_size;
    uint64_t total_tensor_size = 0;
    for (size_t i = 0; i < qcirs.size(); ++i) {
      py::buffer_info info(py::buffer(qcirs[i].cast<py::bytes>()).request());

      uint8_t* qcir_custom_buffer_ptr = static_cast<uint8_t*>(info.ptr);
      QnnQcirCustomProtocol qnn_qcir_custom_protocol;
      auto [status, _, qcir_tensor_size, __, qcir_tensor_ptr] =
          qnn_qcir_custom_protocol.DeserializeQcirCustomBuffer(
              qcir_custom_buffer_ptr);

      if (status != Error::Ok) {
        QNN_EXECUTORCH_LOG_ERROR("Fail to verify QnnQcirCustomProtocol");
        return;
      }

      tensor_ptr.push_back(static_cast<uint8_t*>(qcir_tensor_ptr));
      tensor_size.push_back(qcir_tensor_size);
      total_tensor_size += qcir_tensor_size;
      offsets.push_back(offsets.back() + qcir_tensor_size);
    }

    tensor_data.resize(total_tensor_size);

    // store multiple graphs tensor in a contiguous memory space
    for (size_t i = 0; i < tensor_ptr.size(); ++i) {
      std::memcpy(
          tensor_data.data() + offsets[i], tensor_ptr[i], tensor_size[i]);
    }

    std::vector<flatbuffers::Offset<qcir::Graph>> graphs;
    for (size_t i = 0; i < qcirs.size(); ++i) {
      py::buffer_info info(py::buffer(qcirs[i].cast<py::bytes>()).request());

      uint8_t* qcir_custom_buffer_ptr = static_cast<uint8_t*>(info.ptr);
      QnnQcirCustomProtocol qnn_qcir_custom_protocol;
      auto [status, qcir_fbs_size, _, qcir_fbs_ptr, __] =
          qnn_qcir_custom_protocol.DeserializeQcirCustomBuffer(
              qcir_custom_buffer_ptr);

      if (status != Error::Ok) {
        QNN_EXECUTORCH_LOG_ERROR("Fail to verify QnnQcirCustomProtocol");
        return;
      }

      auto context = qcir::GetContext(qcir_fbs_ptr);
      for (const auto& graph : *context->graphs()) {
        std::vector<flatbuffers::Offset<qcir::Tensor>> tensors;
        for (const auto tensor : *graph->tensors()) {
          // here we need to take a detour to merge multiple qcir flatbuffers
          // outer ToTensor
          //   return: flatbuffers::Offset<Tensor>
          //   consume: QnnTensor, data_offset, flatbuffers::FlatBufferBuilder*
          // inner ToTensor
          //   return: QnnTensor
          //   consume:
          //   flatbuffers::Vector<::flatbuffers::Offset<qcir::Tensor>>,
          //   data_ptr
          tensors.emplace_back(ToTensor(
              ToTensor(tensor, nullptr),
              offsets[i] + tensor->offset(),
              &builder_));
        }
        std::vector<flatbuffers::Offset<qcir::Operator>> nodes;
        for (const auto& node : *graph->nodes()) {
          uint32_t* inputs_ptr = const_cast<uint32_t*>(node->inputs()->data());
          uint32_t* outputs_ptr =
              const_cast<uint32_t*>(node->outputs()->data());
          uint32_t* params_ptr = const_cast<uint32_t*>(node->params()->data());
          std::vector<uint32_t> inputs(
              inputs_ptr, inputs_ptr + node->inputs()->size());
          std::vector<uint32_t> outputs(
              outputs_ptr, outputs_ptr + node->outputs()->size());
          std::vector<uint32_t> params(
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

    // Init QnnQcirCustomProtocol binary
    qnn_executorch_context_binary_ =
        MakeQcirCustomBinaryInfo(qcir_bin, tensor_data);
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

  // this method is specific for stage 2 of compiling multi-graphs
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
      std::vector<uint8_t> tensor_data;
      std::vector<uint64_t> offsets;
      std::unordered_map<void*, int> tensor_map;
      std::vector<flatbuffers::Offset<qcir::Tensor>> fb_tensors;
      std::vector<flatbuffers::Offset<qcir::Operator>> fb_ops;

      auto set_tensor = [&](const std::shared_ptr<TensorWrapper>& wrapper,
                            std::vector<uint32_t>& index) {
        auto it = tensor_map.find(wrapper.get());
        if (it != tensor_map.end()) {
          index.push_back(it->second);
        } else {
          tensor_map[wrapper.get()] = fb_tensors.size();
          index.push_back(fb_tensors.size());
          offsets.push_back(tensor_data.size());
          Qnn_Tensor_t qnn_tensor = wrapper->CloneTensorStruct();
          fb_tensors.emplace_back(
              ToTensor(qnn_tensor, offsets.back(), &builder_));
          uint8_t* data_ptr = static_cast<uint8_t*>(
              QNN_TENSOR_VER_PTR(qnn_tensor)->clientBuf.data);
          if (data_ptr != nullptr) {
            tensor_data.insert(
                tensor_data.end(),
                data_ptr,
                data_ptr + QNN_TENSOR_VER_PTR(qnn_tensor)->clientBuf.dataSize);
          }
        }
      };

      for (std::shared_ptr<OpWrapper>& op_wrapper : op_wrappers) {
        std::vector<uint32_t> inputs, outputs, params;

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
            Qnn_Tensor_t t(
                {.version = QNN_TENSOR_VERSION_2, .v2 = QNN_TENSOR_V2_INIT});
            QNN_TENSOR_VER_PTR(t)->name = p.name;
            QNN_TENSOR_VER_PTR(t)->dataType = p.scalarParam.dataType;
            QNN_TENSOR_VER_PTR(t)->clientBuf.data =
                static_cast<void*>(&p.scalarParam.uint8Value);
            QNN_TENSOR_VER_PTR(t)->clientBuf.dataSize =
                GetDataTypeSize(QNN_TENSOR_VER_PTR(t)->dataType);

            // collect tensor data
            offsets.push_back(tensor_data.size());
            const uint8_t* data_ptr =
                static_cast<uint8_t*>(QNN_TENSOR_VER_PTR(t)->clientBuf.data);
            tensor_data.insert(
                tensor_data.end(),
                data_ptr,
                data_ptr + QNN_TENSOR_VER_PTR(t)->clientBuf.dataSize);
            params.push_back(fb_tensors.size());
            fb_tensors.emplace_back(ToTensor(t, offsets.back(), &builder_));
          }
        }

        Qnn_OpConfig_t op_config = op_wrapper->GetOpConfig();
        fb_ops.emplace_back(qcir::CreateOperatorDirect(
            builder_,
            QNN_OP_VER_PTR(op_config)->name,
            QNN_OP_VER_PTR(op_config)->packageName,
            QNN_OP_VER_PTR(op_config)->typeName,
            &inputs,
            &outputs,
            &params));
      }

      std::vector<flatbuffers::Offset<qcir::Graph>> fb_graphs(
          {qcir::CreateGraphDirect(
              builder_, graph_name.c_str(), &fb_ops, &fb_tensors)});
      auto context = qcir::CreateContextDirect(builder_, &fb_graphs);
      builder_.Finish(context);

      QnnExecuTorchContextBinary qcir_binary(
          {builder_.GetBufferPointer(), builder_.GetSize()});

      custom_qcir_protocol_buffer_ =
          QnnQcirCustomProtocol(qcir_binary.nbytes, tensor_data.size());
      custom_qcir_protocol_buffer_.BuildQcirCustomBuffer(
          qcir_binary, tensor_data);
      std::tie(binary_info.buffer, binary_info.nbytes) =
          custom_qcir_protocol_buffer_.GetCustomProtocolBuffer();
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

  QnnExecuTorchContextBinary MakeQcirCustomBinaryInfo(
      const QnnExecuTorchContextBinary& ctx_bin,
      const std::vector<uint8_t>& tensor_data) {
    custom_qcir_protocol_buffer_ =
        QnnQcirCustomProtocol(ctx_bin.nbytes, tensor_data.size());
    custom_qcir_protocol_buffer_.BuildQcirCustomBuffer(ctx_bin, tensor_data);
    auto [ptr, size] = custom_qcir_protocol_buffer_.GetCustomProtocolBuffer();
    return {ptr, size};
  }

  py::array_t<char> MakeBinaryInfo(const py::bytes& ctx_bin) {
    py::buffer_info info(py::buffer(ctx_bin).request());
    QnnExecuTorchContextBinary binary(
        {info.ptr, static_cast<uint64_t>(info.size * info.itemsize)});

    auto qnn_context_custom_protocol = QnnContextCustomProtocol(binary.nbytes);
    qnn_context_custom_protocol.BuildContextCustomBuffer(binary);
    auto [custom_buffer_ptr, custom_buffer_size] =
        qnn_context_custom_protocol.GetCustomProtocolBuffer();

    auto result = py::array_t<char>(custom_buffer_size);
    auto result_buffer = result.request();
    std::memcpy(result_buffer.ptr, custom_buffer_ptr, custom_buffer_size);
    return result;
  }

  py::array_t<char> StripProtocol(const py::bytes& preprocessed_binary) {
    py::buffer_info info(py::buffer(preprocessed_binary).request());

    void* buf_ptr = nullptr;
    size_t buf_size = 0;
    // check if it's a qnn context binary
    auto [status, signature, ctx_size, ctx_bin] =
        QnnContextCustomProtocol().DeserializeContextCustomBuffer(info.ptr);

    if (status == Error::Ok) {
      buf_size = ctx_size;
      buf_ptr = ctx_bin;
    } else {
      // check if it's a qcir flatbuffers, return fbs if matched
      auto
          [status,
           qcir_fbs_size,
           qcir_tensor_size,
           qcir_fbs_ptr,
           qcir_tensor_ptr] =
              QnnQcirCustomProtocol().DeserializeQcirCustomBuffer(info.ptr);
      if (status == Error::Ok) {
        buf_size = qcir_fbs_size;
        buf_ptr = qcir_fbs_ptr;
      } else {
        // the format should be DLC, return nothing here
        return py::array_t<char>(0);
      }
    }
    auto result = py::array_t<char>(buf_size);
    auto result_buffer = result.request();
    std::memcpy(result_buffer.ptr, buf_ptr, buf_size);
    return result;
  }

 private:
  // Store the bytes object instead of a raw pointer so that this module will
  // keep the bytes alive.
  const py::bytes qnn_executorch_option_ptr_;
  QnnExecuTorchContextBinary qnn_executorch_context_binary_;
  std::shared_ptr<QnnManager> qnn_manager_;
  QnnQcirCustomProtocol custom_qcir_protocol_buffer_;
  QnnContextCustomProtocol custom_context_custom_buffer_;
  flatbuffers::FlatBufferBuilder builder_;
};
} // namespace qnn
} // namespace backends
} // namespace executorch
