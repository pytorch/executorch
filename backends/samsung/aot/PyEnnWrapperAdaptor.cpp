/*
 *  Copyright (c) 2025 Samsung Electronics Co. LTD
 *  All rights reserved
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */

#include <executorch/backends/samsung/aot/PyEnnWrapperAdaptor.h>
#include <pybind11/pybind11.h>

namespace torch {
namespace executor {
namespace enn {
PYBIND11_MODULE(PyEnnWrapperAdaptor, m) {
  pybind11::class_<PyEnnWrapper, std::shared_ptr<PyEnnWrapper>>(m, "EnnWrapper")
      .def(pybind11::init())
      .def("Init", pybind11::overload_cast<>(&PyEnnWrapper::Init))
      .def(
          "Init",
          pybind11::overload_cast<const py::bytes&>(&PyEnnWrapper::Init))
      .def("SetOptions", &PyEnnWrapper::SetOptions)
      .def("IsNodeSupportedByBackend", &PyEnnWrapper::IsNodeSupportedByBackend)
      .def(
          "Compile",
          &PyEnnWrapper::Compile,
          "Ahead of time compilation for serialized graph.")
      .def("GetWeights", &PyEnnWrapper::GetWeights, "Get separated weights.")
      .def("Destroy", &PyEnnWrapper::Destroy, "Release resources.");
}
} // namespace enn
} // namespace executor
} // namespace torch
