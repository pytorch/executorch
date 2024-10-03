/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/aot/python/PyQnnManagerAdaptor.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace torch {
namespace executor {
namespace qnn {
PYBIND11_MODULE(PyQnnManagerAdaptor, m) {
  // TODO: Add related documents for configurations listed below
  using namespace qnn_delegate;

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
      .def("Compile", &PyQnnManager::Compile)
      .def("Destroy", &PyQnnManager::Destroy)
      .def("IsAvailable", &PyQnnManager::IsAvailable)
      .def("IsTensorDump", &PyQnnManager::IsTensorDump)
      .def("AllocateTensor", &PyQnnManager::AllocateTensor)
      .def("GetGraphInputs", &PyQnnManager::GetGraphInputs)
      .def("GetGraphOutputs", &PyQnnManager::GetGraphOutputs)
      .def("GetSpillFillBufferSize", &PyQnnManager::GetSpillFillBufferSize);
}
} // namespace qnn
} // namespace executor
} // namespace torch
