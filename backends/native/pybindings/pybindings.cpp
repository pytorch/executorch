// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <executorch/backends/native/runtime/api/Model.h>
#include <executorch/backends/native/runtime/api/Session.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>
#include <executorch/backends/native/runtime/vulkan/VulkanEngine.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

namespace py = pybind11;

namespace {

py::dtype numpy_dtype(ptn::ScalarType dtype) {
  switch (dtype) {
    case ptn::kByte:
      return py::dtype::of<uint8_t>();
    case ptn::kChar:
      return py::dtype::of<int8_t>();
    case ptn::kShort:
      return py::dtype::of<int16_t>();
    case ptn::kInt:
      return py::dtype::of<int32_t>();
    case ptn::kLong:
      return py::dtype::of<int64_t>();
    case ptn::kHalf:
      return py::dtype("float16");
    case ptn::kFloat:
      return py::dtype::of<float>();
    case ptn::kDouble:
      return py::dtype::of<double>();
    case ptn::kBool:
      return py::dtype::of<bool>();
    case ptn::kUInt16:
      return py::dtype::of<uint16_t>();
    case ptn::kUInt32:
      return py::dtype::of<uint32_t>();
    case ptn::kUInt64:
      return py::dtype::of<uint64_t>();
    case ptn::kBFloat16:
      throw py::type_error(
          "bfloat16 tensors are not supported by NumPy bindings");
  }
  throw py::type_error("unsupported Native tensor dtype");
}

py::array allocate_tensor(const ptn::TensorInfo& info) {
  const auto itemsize = ptn::element_size(info.dtype());
  const auto limit = std::numeric_limits<py::ssize_t>::max();
  if (info.nbytes() > static_cast<size_t>(limit)) {
    throw std::overflow_error("tensor is too large for NumPy");
  }
  std::vector<py::ssize_t> strides;
  for (const auto stride : info.strides()) {
    if (stride > limit / static_cast<py::ssize_t>(itemsize)) {
      throw std::overflow_error("tensor stride is too large for NumPy");
    }
    strides.push_back(stride * itemsize);
  }
  return py::array(
      numpy_dtype(info.dtype()),
      std::vector<py::ssize_t>(info.sizes().begin(), info.sizes().end()),
      strides);
}

std::unique_ptr<ptn::EngineHost> create_host(const std::string& engine) {
  if (engine == "ggml") {
    return ptn::GgmlEngineHost::create();
  }
  if (engine == "vulkan") {
    return ptn::VulkanEngineHost::create();
  }
  throw std::invalid_argument("engine must be 'ggml' or 'vulkan'");
}

class Session final {
 public:
  Session(const ptn::Model& model, const std::string& engine)
      : model_(model),
        host_(create_host(engine)),
        session_(model_.create_session(*host_)) {}

  std::vector<py::array> run(
      const py::sequence& inputs,
      const std::string& method) {
    const auto info = model_.method_info(method);
    if (inputs.size() != info.inputs().size()) {
      throw py::value_error("input count does not match the method");
    }
    std::vector<py::array> input_arrays;
    std::vector<ptn::ConstTensorView> input_views;
    for (size_t i = 0; i < info.inputs().size(); ++i) {
      if (!py::isinstance<py::array>(inputs[i])) {
        throw py::type_error("inputs must be NumPy arrays");
      }
      const auto input = py::cast<py::array>(inputs[i]);
      const auto& spec = info.inputs()[i];
      if (!input.dtype().equal(numpy_dtype(spec.dtype()))) {
        throw py::type_error("input dtype does not match the method");
      }
      if (static_cast<size_t>(input.ndim()) != spec.sizes().size() ||
          !std::equal(
              spec.sizes().begin(), spec.sizes().end(), input.shape())) {
        throw py::value_error("input shape does not match the method");
      }
      // Own input storage in the model's layout before releasing the GIL.
      auto owned = allocate_tensor(spec);
      owned.attr("__setitem__")(py::ellipsis(), input);
      input_views.emplace_back(
          spec.dtype(),
          spec.sizes(),
          spec.strides(),
          owned.data(),
          owned.nbytes());
      input_arrays.push_back(std::move(owned));
    }
    std::vector<py::array> outputs;
    std::vector<ptn::MutableTensorView> output_views;
    for (const auto& spec : info.outputs()) {
      auto output = allocate_tensor(spec);
      output_views.emplace_back(
          spec.dtype(),
          spec.sizes(),
          spec.strides(),
          output.mutable_data(),
          output.nbytes());
      outputs.push_back(std::move(output));
    }
    {
      py::gil_scoped_release release;
      std::lock_guard lock(mutex_);
      session_.run(method, input_views, output_views);
    }
    return outputs;
  }

 private:
  ptn::Model model_;
  std::unique_ptr<ptn::EngineHost> host_;
  ptn::Session session_;
  std::mutex mutex_;
};

} // namespace

PYBIND11_MODULE(_native, module) {
  py::class_<ptn::TensorInfo>(module, "TensorInfo")
      .def_property_readonly(
          "shape",
          [](const ptn::TensorInfo& info) {
            return std::vector<int64_t>(
                info.sizes().begin(), info.sizes().end());
          })
      .def_property_readonly("dtype", [](const ptn::TensorInfo& info) {
        return numpy_dtype(info.dtype());
      });
  py::class_<ptn::MethodInfo>(module, "MethodInfo")
      .def_property_readonly("name", &ptn::MethodInfo::name)
      .def_property_readonly("inputs", &ptn::MethodInfo::inputs)
      .def_property_readonly("outputs", &ptn::MethodInfo::outputs);
  py::class_<Session>(module, "Session")
      .def(
          "run",
          &Session::run,
          py::arg("inputs"),
          py::arg("method") = "forward",
          "Run a method synchronously and return independently owned NumPy outputs.");
  py::class_<ptn::Model>(module, "Model")
      .def_static(
          "load",
          [](const std::filesystem::path& path) {
            return ptn::Model::load_file(path.string());
          },
          py::arg("path"),
          py::call_guard<py::gil_scoped_release>())
      .def_property_readonly(
          "method_names",
          [](const ptn::Model& model) {
            const auto names = model.method_names();
            return std::vector<std::string>(names.begin(), names.end());
          })
      .def("method_info", &ptn::Model::method_info, py::arg("name") = "forward")
      .def(
          "create_session",
          [](const ptn::Model& model, const std::string& engine) {
            return std::make_unique<Session>(model, engine);
          },
          py::arg("engine") = "ggml",
          py::call_guard<py::gil_scoped_release>());
}
