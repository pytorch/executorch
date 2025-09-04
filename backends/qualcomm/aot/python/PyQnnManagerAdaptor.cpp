/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/aot/python/PyQnnManagerAdaptor.h>
#include <pybind11/pybind11.h>
#include "QnnSdkBuildId.h"

namespace py = pybind11;
namespace executorch {
namespace backends {
namespace qnn {

using executorch::runtime::Error;

std::string GetQnnSdkBuildId(std::string library_path) {
  QnnImplementation qnn_loaded_backend = QnnImplementation(library_path);
  ET_CHECK_MSG(
      qnn_loaded_backend.Load(nullptr) == Error::Ok,
      "Fail to load Qnn library");
  const char* id = nullptr;
  // Safe to call any time, backend does not have to be created.
  Qnn_ErrorHandle_t err =
      qnn_loaded_backend.GetQnnInterface().qnn_backend_get_build_id(&id);
  if (err != QNN_SUCCESS || id == nullptr) {
    throw std::runtime_error("Failed to get QNN backend build ID");
  }
  qnn_loaded_backend.TerminateAllBackends();
  return std::string(id);
}

PYBIND11_MODULE(PyQnnManagerAdaptor, m) {
  // TODO: Add related documents for configurations listed below
  using namespace qnn_delegate;

  m.def("GetQnnSdkBuildId", &GetQnnSdkBuildId);
  py::class_<QnnExecuTorchContextBinary>(m, "QnnExecuTorchContextBinary")
      .def(py::init<>());

  py::enum_<Error>(m, "Error")
      .value("Ok", Error::Ok)
      .value("Internal", Error::Internal)
      .export_values();

  py::class_<PyQnnManager, std::shared_ptr<PyQnnManager>>(m, "QnnManager")
      .def(py::init<const py::bytes&>())
      .def(py::init<const py::bytes&, const py::bytes&>())
      .def("Init", &PyQnnManager::Init)
      .def("IsNodeSupportedByBackend", &PyQnnManager::IsNodeSupportedByBackend)
      .def(
          "Compile",
          py::overload_cast<
              const std::vector<std::string>&,
              std::vector<std::vector<std::shared_ptr<OpWrapper>>>&>(
              &PyQnnManager::Compile))
      .def("Destroy", &PyQnnManager::Destroy)
      .def("IsAvailable", &PyQnnManager::IsAvailable)
      .def("IsTensorDump", &PyQnnManager::IsTensorDump)
      .def("AllocateTensor", &PyQnnManager::AllocateTensor)
      .def("GetGraphInputs", &PyQnnManager::GetGraphInputs)
      .def("GetGraphOutputs", &PyQnnManager::GetGraphOutputs)
      .def("GetGraphNames", &PyQnnManager::GetGraphNames)
      .def("GetSpillFillBufferSize", &PyQnnManager::GetSpillFillBufferSize)
      .def(
          "MakeBinaryInfo",
          py::overload_cast<const py::bytes&>(&PyQnnManager::MakeBinaryInfo))
      .def("StripProtocol", &PyQnnManager::StripProtocol);
}
} // namespace qnn
} // namespace backends
} // namespace executorch
