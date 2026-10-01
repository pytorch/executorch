/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

namespace executorch::extension::pybindings {

namespace py = pybind11;

/** Owned tensor result returned by portable Python bindings. */
class PyExecuTorchResult final {
 public:
  explicit PyExecuTorchResult(const executorch::aten::Tensor& tensor)
      : storage_(tensor.nbytes()),
        sizes_(tensor.sizes().begin(), tensor.sizes().end()),
        strides_(tensor.strides().begin(), tensor.strides().end()),
        scalar_type_(tensor.scalar_type()) {
    if (!tensor.device().is_cpu()) {
      throw std::runtime_error(
          "ExecuTorch results only support CPU outputs until DLPack is enabled");
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
    const auto ndim = shape.size();
    return py::buffer_info(
        storage_.data(),
        itemsize,
        buffer_format(scalar_type_),
        ndim,
        std::move(shape),
        std::move(byte_strides),
        /*readonly=*/false);
  }

  py::buffer_info flat_buffer() {
    const auto itemsize = executorch::runtime::elementSize(scalar_type_);
    return py::buffer_info(
        storage_.data(),
        itemsize,
        buffer_format(scalar_type_),
        /*ndim=*/1,
        {static_cast<py::ssize_t>(storage_.size() / itemsize)},
        {static_cast<py::ssize_t>(itemsize)},
        /*readonly=*/false);
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

  py::tuple element_strides() const {
    py::tuple result(strides_.size());
    for (size_t i = 0; i < strides_.size(); ++i) {
      result[i] = strides_[i];
    }
    return result;
  }

  py::dtype dtype() const {
    return py::dtype(numpy_dtype_name(scalar_type_));
  }

  size_t nbytes() const {
    return storage_.size();
  }

  const char* torch_dtype_name() const {
    using executorch::aten::ScalarType;
    switch (scalar_type_) {
      case ScalarType::Byte:
        return "uint8";
      case ScalarType::Char:
        return "int8";
      case ScalarType::Short:
        return "int16";
      case ScalarType::Int:
        return "int32";
      case ScalarType::Long:
        return "int64";
      case ScalarType::Half:
        return "float16";
      case ScalarType::Float:
        return "float32";
      case ScalarType::Double:
        return "float64";
      case ScalarType::ComplexFloat:
        return "complex64";
      case ScalarType::ComplexDouble:
        return "complex128";
      case ScalarType::Bool:
        return "bool";
      case ScalarType::BFloat16:
        return "bfloat16";
      case ScalarType::UInt16:
        return "uint16";
      case ScalarType::UInt32:
        return "uint32";
      case ScalarType::UInt64:
        return "uint64";
      default:
        throw std::runtime_error(
            "ExecuTorch result dtype cannot be represented by PyTorch");
    }
  }

 private:
  static const char* numpy_dtype_name(executorch::aten::ScalarType type) {
    using executorch::aten::ScalarType;
    switch (type) {
      case ScalarType::Byte:
        return "uint8";
      case ScalarType::Char:
        return "int8";
      case ScalarType::Short:
        return "int16";
      case ScalarType::Int:
        return "int32";
      case ScalarType::Long:
        return "int64";
      case ScalarType::Half:
        return "float16";
      case ScalarType::Float:
        return "float32";
      case ScalarType::Double:
        return "float64";
      case ScalarType::ComplexFloat:
        return "complex64";
      case ScalarType::ComplexDouble:
        return "complex128";
      case ScalarType::Bool:
        return "bool";
      case ScalarType::BFloat16:
      case ScalarType::UInt16:
        return "uint16";
      case ScalarType::UInt32:
        return "uint32";
      case ScalarType::UInt64:
        return "uint64";
      default:
        throw std::runtime_error(
            "ExecuTorch result dtype cannot be represented by NumPy");
    }
  }

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

/** Contiguous buffer facade used to construct a strided torch.Tensor result. */
class PyExecuTorchResultFlatBuffer final {
 public:
  explicit PyExecuTorchResultFlatBuffer(
      std::shared_ptr<PyExecuTorchResult> result)
      : result_(std::move(result)) {}

  py::buffer_info buffer() {
    return result_->flat_buffer();
  }

 private:
  std::shared_ptr<PyExecuTorchResult> result_;
};

} // namespace executorch::extension::pybindings
