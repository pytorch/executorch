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
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

namespace executorch::extension::pybindings {

namespace py = pybind11;

struct PyTensorDType final {
  executorch::aten::ScalarType scalar_type;
  const char* torch_name;
  const char* numpy_name;
  const char* buffer_format;
};

inline const PyTensorDType& python_dtype(
    executorch::aten::ScalarType scalar_type) {
  using executorch::aten::ScalarType;
  static constexpr PyTensorDType kDTypes[] = {
      {ScalarType::Byte, "uint8", "uint8", "B"},
      {ScalarType::Char, "int8", "int8", "b"},
      {ScalarType::Short, "int16", "int16", "h"},
      {ScalarType::Int, "int32", "int32", "i"},
      {ScalarType::Long, "int64", "int64", "q"},
      {ScalarType::Half, "float16", "float16", "e"},
      {ScalarType::Float, "float32", "float32", "f"},
      {ScalarType::Double, "float64", "float64", "d"},
      {ScalarType::ComplexHalf, "complex32", nullptr, nullptr},
      {ScalarType::ComplexFloat, "complex64", "complex64", "Zf"},
      {ScalarType::ComplexDouble, "complex128", "complex128", "Zd"},
      {ScalarType::Bool, "bool", "bool", "?"},
      {ScalarType::QInt8, "qint8", nullptr, nullptr},
      {ScalarType::QUInt8, "quint8", nullptr, nullptr},
      {ScalarType::QInt32, "qint32", nullptr, nullptr},
      {ScalarType::BFloat16, "bfloat16", nullptr, nullptr},
      {ScalarType::QUInt4x2, "quint4x2", nullptr, nullptr},
      {ScalarType::QUInt2x4, "quint2x4", nullptr, nullptr},
      {ScalarType::Bits1x8, "bits1x8", nullptr, nullptr},
      {ScalarType::Bits2x4, "bits2x4", nullptr, nullptr},
      {ScalarType::Bits4x2, "bits4x2", nullptr, nullptr},
      {ScalarType::Bits8, "bits8", nullptr, nullptr},
      {ScalarType::Bits16, "bits16", nullptr, nullptr},
      {ScalarType::Float8_e5m2, "float8_e5m2", nullptr, nullptr},
      {ScalarType::Float8_e4m3fn, "float8_e4m3fn", nullptr, nullptr},
      {ScalarType::Float8_e5m2fnuz, "float8_e5m2fnuz", nullptr, nullptr},
      {ScalarType::Float8_e4m3fnuz, "float8_e4m3fnuz", nullptr, nullptr},
      {ScalarType::UInt16, "uint16", "uint16", "H"},
      {ScalarType::UInt32, "uint32", "uint32", "I"},
      {ScalarType::UInt64, "uint64", "uint64", "Q"},
  };
  const auto index = static_cast<size_t>(scalar_type);
  if (index >= sizeof(kDTypes) / sizeof(kDTypes[0]) ||
      kDTypes[index].scalar_type != scalar_type) {
    throw std::runtime_error("Unsupported ExecuTorch result dtype");
  }
  return kDTypes[index];
}

inline executorch::aten::ScalarType scalar_type_from_torch_dtype(
    const std::string& torch_dtype) {
  using executorch::aten::ScalarType;
  for (int8_t value = 0; value < static_cast<int8_t>(ScalarType::NumOptions);
       ++value) {
    const auto scalar_type = static_cast<ScalarType>(value);
    const auto& dtype = python_dtype(scalar_type);
    if (torch_dtype == std::string("torch.") + dtype.torch_name) {
      return scalar_type;
    }
  }
  throw py::value_error("Unsupported torch dtype: " + torch_dtype);
}

/** Tensor result returned by portable Python bindings. */
class PyExecuTorchResult final {
 public:
  explicit PyExecuTorchResult(
      const executorch::aten::Tensor& tensor,
      bool clone = true,
      py::object owner = py::none())
      : sizes_(tensor.sizes().begin(), tensor.sizes().end()),
        strides_(tensor.strides().begin(), tensor.strides().end()),
        scalar_type_(tensor.scalar_type()),
        nbytes_(tensor.nbytes()),
        owner_(std::move(owner)) {
    if (!tensor.device().is_cpu()) {
      throw std::runtime_error(
          "ExecuTorch results only support CPU outputs until DLPack is enabled");
    }
    if (clone) {
      storage_.resize(nbytes_);
      if (!storage_.empty()) {
        std::memcpy(storage_.data(), tensor.const_data_ptr(), storage_.size());
      }
      data_ = storage_.data();
      owner_ = py::none();
    } else {
      data_ = const_cast<void*>(tensor.const_data_ptr());
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
        data_,
        itemsize,
        require_buffer_format(),
        ndim,
        std::move(shape),
        std::move(byte_strides),
        /*readonly=*/false);
  }

  py::buffer_info flat_buffer() {
    // torch.frombuffer receives the real dtype separately. Export raw bytes so
    // this facade also works for dtypes that PEP 3118 cannot describe.
    return py::buffer_info(
        data_,
        /*itemsize=*/1,
        /*format=*/"B",
        /*ndim=*/1,
        {static_cast<py::ssize_t>(nbytes_)},
        {1},
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
    const auto* name = python_dtype(scalar_type_).numpy_name;
    if (name == nullptr) {
      throw std::runtime_error(
          std::string("ExecuTorch result dtype ") +
          executorch::runtime::toString(scalar_type_) +
          " cannot be represented by NumPy");
    }
    return py::dtype(name);
  }

  py::array numpy_array(const py::object& base) {
    const auto array_dtype = dtype();
    auto info = buffer();
    return py::array(array_dtype, info.shape, info.strides, info.ptr, base);
  }

  size_t nbytes() const {
    return nbytes_;
  }

  const char* torch_dtype_name() const {
    return python_dtype(scalar_type_).torch_name;
  }

 private:
  const char* require_buffer_format() const {
    const auto* format = python_dtype(scalar_type_).buffer_format;
    if (format == nullptr) {
      throw py::buffer_error(
          std::string("ExecuTorch result dtype ") +
          executorch::runtime::toString(scalar_type_) +
          " cannot use the Python buffer protocol");
    }
    return format;
  }

  void* data_ = nullptr;
  std::vector<uint8_t> storage_;
  std::vector<executorch::aten::SizesType> sizes_;
  std::vector<executorch::aten::StridesType> strides_;
  executorch::aten::ScalarType scalar_type_;
  size_t nbytes_;
  py::object owner_;
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
