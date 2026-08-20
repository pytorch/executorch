/*
 *  Copyright (c) 2025 Samsung Electronics Co. LTD
 *  All rights reserved
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree.
 *
 */

#include "PyGraphWrapperAdaptor.h"

namespace torch {
namespace executor {
namespace enn {

PYBIND11_MODULE(PyGraphWrapperAdaptor, m) {
  pybind11::class_<OpParamWrapper, std::shared_ptr<OpParamWrapper>>(
      m, "OpParamWrapper")
      .def(pybind11::init<std::string>())
      .def("SetStringValue", &OpParamWrapper::SetStringValue)
      .def("SetScalarValue", &OpParamWrapper::SetScalarValue<double>)
      .def("SetScalarValue", &OpParamWrapper::SetScalarValue<float>)
      .def("SetScalarValue", &OpParamWrapper::SetScalarValue<bool>)
      .def("SetScalarValue", &OpParamWrapper::SetScalarValue<uint32_t>)
      .def("SetScalarValue", &OpParamWrapper::SetScalarValue<int32_t>)
      .def("SetScalarValue", &OpParamWrapper::SetScalarValue<uint64_t>)
      .def("SetScalarValue", &OpParamWrapper::SetScalarValue<int64_t>)
      .def("SetVectorValue", &OpParamWrapper::SetVectorValue<double>)
      .def("SetVectorValue", &OpParamWrapper::SetVectorValue<float>)
      .def("SetVectorValue", &OpParamWrapper::SetVectorValue<uint32_t>)
      .def("SetVectorValue", &OpParamWrapper::SetVectorValue<int32_t>)
      .def("SetVectorValue", &OpParamWrapper::SetVectorValue<uint64_t>)
      .def("SetVectorValue", &OpParamWrapper::SetVectorValue<int64_t>);

  pybind11::class_<EnnTensorWrapper, std::shared_ptr<EnnTensorWrapper>>(
      m, "PyEnnTensorWrapper")
      .def(pybind11::init<
           std::string,
           const std::vector<DIM_T>&,
           std::string,
           std::string>())
      .def(
          "AddQuantizeParam",
          &EnnTensorWrapper::AddQuantizeParam,
          "Add quantize parameter.")
      .def(
          "AddData",
          &EnnTensorWrapper::AddData,
          "Add data for constant tensor.");

  pybind11::class_<EnnOpWrapper, std::shared_ptr<EnnOpWrapper>>(
      m, "PyEnnOpWrapper")
      .def(pybind11::init<
           std::string,
           std::string,
           const std::vector<TENSOR_ID_T>&,
           const std::vector<TENSOR_ID_T>&>())
      .def(
          "AddOpParam",
          &EnnOpWrapper::AddOpParam,
          "Add parameter for current op.");

  pybind11::class_<PyEnnGraphWrapper, std::shared_ptr<PyEnnGraphWrapper>>(
      m, "PyEnnGraphWrapper")
      .def(pybind11::init())
      .def("Init", &PyEnnGraphWrapper::Init, "Initialize Graph Wrapper.")
      .def(
          "DefineTensor",
          &PyEnnGraphWrapper::DefineTensor,
          "Define a tensor in graph.")
      .def(
          "DefineOpNode",
          &PyEnnGraphWrapper::DefineOpNode,
          "Define a op node in graph.")
      .def(
          "SetGraphInputTensors",
          &PyEnnGraphWrapper::SetGraphInputTensors,
          "Set inputs for Graph")
      .def(
          "SetGraphOutputTensors",
          &PyEnnGraphWrapper::SetGraphOutputTensors,
          "Set outputs for Graph")
      .def(
          "FinishBuild",
          &PyEnnGraphWrapper::FinishBuild,
          "Finish to build the graph.")
      .def("Serialize", &PyEnnGraphWrapper::Serialize, "Serialize the graph.");
}

} // namespace enn
} // namespace executor
} // namespace torch
