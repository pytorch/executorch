/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>

#include <ATen/Tensor.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <torch/csrc/utils/pybind.h>
#include "executorch/extension/tensor/tensor.h"
#include "executorch/extension/training/optimizer/sgd.h"
#ifndef USE_ATEN_LIB
#include <executorch/extension/aten_util/aten_bridge.h>
#endif

namespace py = pybind11;

namespace executorch {
namespace extension {
namespace training {

namespace {

struct PySGD final {
  explicit PySGD(
      const py::dict& named_params,
      double lr,
      double momentum,
      double dampening,
      double weight_decay,
      bool nesterov)
      : sgd_(nullptr),
        fqns_()
#ifndef USE_ATEN_LIB
        ,
        params_()
#endif
  {
    std::map<std::string_view, executorch::aten::Tensor> cpp_inputs;
    auto py_named_params =
        py::cast<std::unordered_map<std::string, at::Tensor>>(named_params);
    const auto params_size = py::len(named_params);
    fqns_ = std::vector<std::string>();
    fqns_.reserve(params_size);

    for (auto pair : py_named_params) {
      fqns_.push_back(pair.first);
      std::string_view v{fqns_.back().c_str(), pair.first.size()};
#ifndef USE_ATEN_LIB
      // convert at::Tensor to torch::executor::Tensor
      params_.emplace_back(alias_tensor_ptr_to_attensor(pair.second));
      cpp_inputs.insert({v, *params_.back()});
#else
      cpp_inputs.insert({v, pair.second});
#endif
    }
    sgd_ = std::make_unique<optimizer::SGD>(
        cpp_inputs,
        extension::training::optimizer::SGDOptions(
            lr, momentum, dampening, weight_decay, nesterov));
  }

  // Not needed for now, so just delete.
  PySGD(const PySGD&) = delete;
  PySGD& operator=(const PySGD&) = delete;
  PySGD(PySGD&&) = delete;
  PySGD& operator=(PySGD&&) = delete;

  void step(const py::dict& py_dict) {
    auto py_named_gradients =
        py::cast<std::unordered_map<std::string, at::Tensor>>(py_dict);
    std::map<std::string_view, executorch::aten::Tensor> cpp_inputs;

    std::vector<std::string> fqn;
#ifndef USE_ATEN_LIB
    std::vector<TensorPtr> et_tensors;
#endif

    // Convert python objects into cpp.
    for (const auto& pair : py_named_gradients) {
      fqn.push_back(pair.first);
      auto at_tensor = pair.second;
      // alias_etensor_to_attensor will assert on this later, so to better
      // propogate up to python we check early and throw an exception.
      if (!at_tensor.is_contiguous()) {
        auto error_msg = "Gradient is not contiguous.";
        throw std::runtime_error(error_msg);
      }
#ifndef USE_ATEN_LIB
      // convert at::Tensor to torch::executor::Tensor
      auto temp = alias_tensor_ptr_to_attensor(at_tensor);
      et_tensors.push_back(temp);
      cpp_inputs.insert({pair.first.c_str(), *et_tensors.back()});
#else
      cpp_inputs.insert({pair.first.c_str(), at_tensor});
#endif
    }

    auto err = sgd_->step(cpp_inputs);
    if (err != runtime::Error::Ok) {
      throw std::runtime_error("SGD step failed");
    }
  }

 private:
  // TODO(jakeszwe): Write an optimizer interface and use it here instead of SGD
  // specifically.
  std::unique_ptr<optimizer::SGD> sgd_ = nullptr;
  std::vector<std::string> fqns_;

#ifndef USE_ATEN_LIB // Portable mode
  std::vector<TensorPtr> params_;
#endif
  ;
};

static std::unique_ptr<PySGD> get_sgd_optimizer(
    const py::dict& named_params,
    double lr,
    double momentum = 0,
    double dampening = 0,
    double weight_decay = 0,
    bool nesterov = false) {
  return std::make_unique<PySGD>(
      named_params, lr, momentum, dampening, weight_decay, nesterov);
}

} // namespace

PYBIND11_MODULE(_training_lib, m) {
  m.def(
      "get_sgd_optimizer",
      &get_sgd_optimizer,
      py::arg("named_params"),
      py::arg("lr") = 0.1,
      py::arg("momentum") = 0.0,
      py::arg("dampening") = 0.0,
      py::arg("weight_decay") = 0.0,
      py::arg("nesterov") = false);
  py::class_<PySGD>(m, "ExecuTorchSGD").def("step", &PySGD::step);
}

} // namespace training
} // namespace extension
} // namespace executorch
