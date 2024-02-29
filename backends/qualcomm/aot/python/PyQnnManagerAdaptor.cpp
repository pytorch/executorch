/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/aot/ir/qcir_utils.h>
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/backends/qualcomm/runtime/QnnManager.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace torch {
namespace executor {
namespace qnn {
PYBIND11_MODULE(PyQnnManagerAdaptor, m) {
  // TODO: Add related documents for configurations listed below

  m.def("QnnExecuTorchOptionsDefault", &QnnExecuTorchOptionsDefault);
  py::enum_<QnnExecuTorchBackendType>(m, "QnnExecuTorchBackendType")
      .value("kUndefinedBackend", QnnExecuTorchBackendType::kUndefinedBackend)
      .value("kGpuBackend", QnnExecuTorchBackendType::kGpuBackend)
      .value("kHtpBackend", QnnExecuTorchBackendType::kHtpBackend)
      .value("kDspBackend", QnnExecuTorchBackendType::kDspBackend)
      .export_values();

  py::enum_<QnnExecuTorchLogLevel>(m, "QnnExecuTorchLogLevel")
      .value("kLogOff", QnnExecuTorchLogLevel::kLogOff)
      .value("kLogLevelError", QnnExecuTorchLogLevel::kLogLevelError)
      .value("kLogLevelWarn", QnnExecuTorchLogLevel::kLogLevelWarn)
      .value("kLogLevelInfo", QnnExecuTorchLogLevel::kLogLevelInfo)
      .value("kLogLevelVerbose", QnnExecuTorchLogLevel::kLogLevelVerbose)
      .value("kLogLevelDebug", QnnExecuTorchLogLevel::kLogLevelDebug)
      .export_values();

  py::enum_<QcomChipset>(m, "QcomChipset")
      .value("UNKNOWN_SM", QcomChipset::UNKNOWN_SM)
      .value("SM8450", QcomChipset::SM8450)
      .value("SM8475", QcomChipset::SM8475)
      .value("SM8550", QcomChipset::SM8550)
      .value("SM8650", QcomChipset::SM8650)
      .export_values();

  py::enum_<QnnExecuTorchHtpPrecision>(m, "QnnExecuTorchHtpPrecision")
      .value("kHtpQuantized", QnnExecuTorchHtpPrecision::kHtpQuantized)
      .value("kHtpFp16", QnnExecuTorchHtpPrecision::kHtpFp16)
      .export_values();

  py::enum_<QnnExecuTorchHtpPdSession>(m, "QnnExecuTorchHtpPdSession")
      .value("kHtpUnsignedPd", QnnExecuTorchHtpPdSession::kHtpUnsignedPd)
      .value("kHtpSignedPd", QnnExecuTorchHtpPdSession::kHtpSignedPd)
      .export_values();

  py::enum_<QnnExecuTorchHtpPerformanceMode>(
      m, "QnnExecuTorchHtpPerformanceMode")
      .value("kHtpDefault", QnnExecuTorchHtpPerformanceMode::kHtpDefault)
      .value(
          "kHtpSustainedHighPerformance",
          QnnExecuTorchHtpPerformanceMode::kHtpSustainedHighPerformance)
      .value("kHtpBurst", QnnExecuTorchHtpPerformanceMode::kHtpBurst)
      .value(
          "kHtpHighPerformance",
          QnnExecuTorchHtpPerformanceMode::kHtpHighPerformance)
      .value("kHtpPowerSaver", QnnExecuTorchHtpPerformanceMode::kHtpPowerSaver)
      .value(
          "kHtpLowPowerSaver",
          QnnExecuTorchHtpPerformanceMode::kHtpLowPowerSaver)
      .value(
          "kHtpHighPowerSaver",
          QnnExecuTorchHtpPerformanceMode::kHtpHighPowerSaver)
      .value(
          "kHtpLowBalanced", QnnExecuTorchHtpPerformanceMode::kHtpLowBalanced)
      .value("kHtpBalanced", QnnExecuTorchHtpPerformanceMode::kHtpBalanced)
      .export_values();

  py::class_<QnnExecuTorchHtpBackendOptions>(
      m, "QnnExecuTorchHtpBackendOptions")
      .def(py::init<>())
      .def_readwrite("soc_model", &QnnExecuTorchHtpBackendOptions::soc_model)
      .def_readwrite("precision", &QnnExecuTorchHtpBackendOptions::precision)
      .def_readwrite(
          "performance_mode", &QnnExecuTorchHtpBackendOptions::performance_mode)
      .def_readwrite("pd_session", &QnnExecuTorchHtpBackendOptions::pd_session)
      .def_readwrite(
          "use_conv_hmx", &QnnExecuTorchHtpBackendOptions::use_conv_hmx)
      .def_readwrite(
          "use_fold_relu", &QnnExecuTorchHtpBackendOptions::use_fold_relu);

  py::class_<QnnExecuTorchContextBinary>(m, "QnnExecuTorchContextBinary")
      .def(py::init<>());

  py::class_<QnnExecuTorchOptions>(m, "QnnExecuTorchOptions")
      .def(py::init<>())
      .def_readwrite("backend_type", &QnnExecuTorchOptions::backend_type)
      .def_readwrite("htp_options", &QnnExecuTorchOptions::htp_options)
      .def_readwrite("log_level", &QnnExecuTorchOptions::log_level)
      .def_readwrite("online_prepare", &QnnExecuTorchOptions::online_prepare)
      .def_property(
          "library_path",
          [](const QnnExecuTorchOptions& self) -> py::str {
            return self.library_path;
          },
          [](QnnExecuTorchOptions& self, const std::string& new_library_path) {
            self.library_path = strdup(new_library_path.data());
          })
      .def_property(
          "graph_name",
          [](const QnnExecuTorchOptions& self) -> py::str {
            return self.graph_name;
          },
          [](QnnExecuTorchOptions& self, const std::string& new_graph_name) {
            self.graph_name = strdup(new_graph_name.data());
          });

  py::enum_<Error>(m, "Error")
      .value("Ok", Error::Ok)
      .value("Internal", Error::Internal)
      .export_values();

  py::class_<QnnManager>(m, "QnnManager")
      .def(py::init<const QnnExecuTorchOptions*>())
      .def("Init", &QnnManager::Init)
      .def("IsNodeSupportedByBackend", &QnnManager::IsNodeSupportedByBackend)
      .def(
          "Compile",
          [](QnnManager& self,
             std::vector<std::shared_ptr<OpWrapper>>& op_wrappers) {
            QnnExecuTorchContextBinary context_binary;
            flatbuffers::FlatBufferBuilder builder;

            if (self.IsOnlinePrepare()) {
              std::vector<flatbuffers::Offset<qcir::Tensor>> tensors;
              std::unordered_map<void*, int> tensor_map;

              auto set_tensor =
                  [&](const std::shared_ptr<TensorWrapper>& wrapper,
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

                for (const auto& tensor_wrapper :
                     op_wrapper->GetInputTensors()) {
                  set_tensor(tensor_wrapper, inputs);
                }

                for (const auto& tensor_wrapper :
                     op_wrapper->GetOutputTensors()) {
                  set_tensor(tensor_wrapper, outputs);
                }

                for (const auto& param : op_wrapper->GetParams()) {
                  auto* p_tensor_param =
                      dynamic_cast<TensorParamWrapper*>(param.get());
                  if (p_tensor_param != nullptr) {
                    auto wrapper = p_tensor_param->GetTensorWrapper();
                    wrapper->SetName(param->GetName());
                    set_tensor(wrapper, params);
                  } else {
                    Error err = param->PopulateQnnParam();
                    if (err != Error::Ok) {
                      QNN_EXECUTORCH_LOG(
                          kLogLevelError,
                          "[Qnn ExecuTorch] Fail to get scalar parameter in online prepare stage");
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
              auto graph =
                  qcir::CreateGraphDirect(builder, &operators, &tensors);
              builder.Finish(graph);
              context_binary.buffer = builder.GetBufferPointer();
              context_binary.nbytes = builder.GetSize();
            } else if (self.Compile(op_wrappers, context_binary) != Error::Ok) {
              return py::array_t<char>(0);
            }

            // allocate py::array (to pass the result of the C++ function to
            // Python)
            auto result = py::array_t<char>(context_binary.nbytes);
            auto result_buffer = result.request();
            char* result_ptr = (char*)result_buffer.ptr;
            std::memcpy(
                result_ptr, context_binary.buffer, context_binary.nbytes);
            return result;
          })
      .def("Destroy", &QnnManager::Destroy)
      .def("IsAvailable", &QnnManager::IsAvailable);
}
} // namespace qnn
} // namespace executor
} // namespace torch
