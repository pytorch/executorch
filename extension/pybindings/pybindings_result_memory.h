/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstring>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

namespace executorch::extension::pybindings {

namespace py = pybind11;

/** Read-only owned memory returned by torch-free Python bindings. */
class PyResultMemory final {
 public:
  explicit PyResultMemory(const executorch::aten::Tensor& tensor)
      : storage_(tensor.nbytes()),
        sizes_(tensor.sizes().begin(), tensor.sizes().end()),
        strides_(tensor.strides().begin(), tensor.strides().end()),
        scalar_type_(tensor.scalar_type()) {
    if (!tensor.device().is_cpu()) {
      throw std::runtime_error(
          "Result memory only supports CPU outputs until DLPack is enabled");
    }
    if (!storage_.empty()) {
      std::memcpy(storage_.data(), tensor.const_data_ptr(), storage_.size());
    }
  }

  py::buffer_info buffer() {
    std::vector<py::ssize_t> shape(sizes_.begin(), sizes_.end());
    std::vector<py::ssize_t> byte_strides;
    const auto itemsize = executorch::runtime::elementSize(scalar_type_);
    byte_strides.reserve(strides_.size());
    for (const auto stride : strides_) {
      byte_strides.push_back(stride * itemsize);
    }
    return py::buffer_info(
        storage_.data(),
        itemsize,
        buffer_format(scalar_type_),
        shape.size(),
        std::move(shape),
        std::move(byte_strides),
        /*readonly=*/true);
  }

  py::tuple shape() const {
    py::tuple result(sizes_.size());
    for (size_t i = 0; i < sizes_.size(); ++i) {
      result[i] = sizes_[i];
    }
    return result;
  }

  py::tuple strides() const {
    py::tuple result(strides_.size());
    const auto itemsize = executorch::runtime::elementSize(scalar_type_);
    for (size_t i = 0; i < strides_.size(); ++i) {
      result[i] = strides_[i] * itemsize;
    }
    return result;
  }

  py::dtype dtype() const {
    return py::dtype(buffer_format(scalar_type_));
  }

  size_t nbytes() const {
    return storage_.size();
  }

 private:
  static const char* buffer_format(executorch::aten::ScalarType type) {
    using executorch::aten::ScalarType;
    switch (type) {
      case ScalarType::Byte:
        return "B";
      case ScalarType::Char:
        return "b";
      case ScalarType::Short:
        return "h";
      case ScalarType::Int:
        return "i";
      case ScalarType::Long:
        return "q";
      case ScalarType::Half:
        return "e";
      case ScalarType::Float:
        return "f";
      case ScalarType::Double:
        return "d";
      case ScalarType::ComplexFloat:
        return "Zf";
      case ScalarType::ComplexDouble:
        return "Zd";
      case ScalarType::Bool:
        return "?";
      case ScalarType::BFloat16:
      case ScalarType::UInt16:
        return "H";
      case ScalarType::UInt32:
        return "I";
      case ScalarType::UInt64:
        return "Q";
      default:
        throw std::runtime_error(
            "ExecuTorch result dtype cannot use the Python buffer protocol");
    }
  }

  std::vector<uint8_t> storage_;
  std::vector<executorch::aten::SizesType> sizes_;
  std::vector<executorch::aten::StridesType> strides_;
  executorch::aten::ScalarType scalar_type_;
};

} // namespace executorch::extension::pybindings
