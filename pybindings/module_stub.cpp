// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

namespace torch {
namespace executor {

void init_module_functions(py::module_& m) {}

} // namespace executor
} // namespace torch
