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

#include <executorch/extension/pybindings/pybindings_dlpack.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

namespace executorch::extension::pybindings {

namespace py = pybind11;

/** Owned tensor result returned by portable Python bindings. */
class PyExecuTorchResult final
    : public std::enable_shared_from_this<PyExecuTorchResult> {
 public:
  explicit PyExecuTorchResult(const executorch::aten::Tensor& tensor)
      : storage_(tensor.nbytes()),
        sizes_(tensor.sizes().begin(), tensor.sizes().end()),
        strides_(tensor.strides().begin(), tensor.strides().end()),
        scalar_type_(tensor.scalar_type()),
        device_(tensor.device()) {
    if (!tensor.device().is_cpu()) {
      throw std::runtime_error(
          "ExecuTorch results only support CPU outputs until DLPack is enabled");
    }
    if (!storage_.empty()) {
      std::memcpy(storage_.data(), tensor.const_data_ptr(), storage_.size());
    }
    data_ = storage_.data();
  }

  PyExecuTorchResult(
      const executorch::aten::Tensor& tensor,
      std::shared_ptr<void> owner)
      : data_(const_cast<void*>(tensor.const_data_ptr())),
        nbytes_(tensor.nbytes()),
        sizes_(tensor.sizes().begin(), tensor.sizes().end()),
        strides_(tensor.strides().begin(), tensor.strides().end()),
        scalar_type_(tensor.scalar_type()),
        device_(tensor.device()),
        owner_(std::move(owner)) {
    if (nbytes_ != 0 && data_ == nullptr) {
      throw std::runtime_error("ExecuTorch result data is not allocated");
    }
  }

  py::buffer_info buffer() {
    if (!device_.is_cpu()) {
      throw py::buffer_error(
          "Device ExecuTorch results are available through DLPack, not the buffer protocol");
    }
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
        buffer_format(scalar_type_),
        ndim,
        std::move(shape),
        std::move(byte_strides),
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
    return storage_.empty() ? nbytes_ : storage_.size();
  }

  py::tuple dlpack_device() const {
    return py::make_tuple(
        static_cast<int>(
            device_.is_cpu() ? dlpack::DeviceType::CPU
                             : dlpack::DeviceType::CUDA),
        static_cast<int>(device_.index()));
  }

  py::capsule to_dlpack(const py::object& stream = py::none()) {
    (void)stream;
    auto* holder = new DLPackHolder(shared_from_this());
    return py::capsule(&holder->managed, "dltensor", [](PyObject* capsule) {
      if (PyCapsule_IsValid(capsule, "dltensor")) {
        auto* managed = static_cast<dlpack::ManagedTensor*>(
            PyCapsule_GetPointer(capsule, "dltensor"));
        managed->deleter(managed);
      }
    });
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

  struct DLPackHolder final {
    explicit DLPackHolder(std::shared_ptr<PyExecuTorchResult> result)
        : result(std::move(result)),
          shape(this->result->sizes_.begin(), this->result->sizes_.end()),
          strides(
              this->result->strides_.begin(),
              this->result->strides_.end()) {
      managed.dl_tensor.data = this->result->data_;
      managed.dl_tensor.device = this->result->dl_device();
      managed.dl_tensor.ndim = static_cast<int32_t>(shape.size());
      managed.dl_tensor.dtype = this->result->dl_dtype();
      managed.dl_tensor.shape = shape.data();
      managed.dl_tensor.strides = strides.data();
      managed.dl_tensor.byte_offset = 0;
      managed.manager_ctx = this;
      managed.deleter = [](dlpack::ManagedTensor* self) {
        delete static_cast<DLPackHolder*>(self->manager_ctx);
      };
    }

    std::shared_ptr<PyExecuTorchResult> result;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    dlpack::ManagedTensor managed{};
  };

  dlpack::Device dl_device() const {
    if (device_.is_cpu()) {
      return {dlpack::DeviceType::CPU, device_.index()};
    }
    if (device_.type() == executorch::aten::DeviceType::CUDA) {
      return {dlpack::DeviceType::CUDA, device_.index()};
    }
    throw std::runtime_error("Result device is not supported by DLPack");
  }

  dlpack::DataType dl_dtype() const {
    using executorch::aten::ScalarType;
    switch (scalar_type_) {
      case ScalarType::Byte:
        return {dlpack::DataTypeCode::UInt, 8, 1};
      case ScalarType::Char:
        return {dlpack::DataTypeCode::Int, 8, 1};
      case ScalarType::Short:
        return {dlpack::DataTypeCode::Int, 16, 1};
      case ScalarType::Int:
        return {dlpack::DataTypeCode::Int, 32, 1};
      case ScalarType::Long:
        return {dlpack::DataTypeCode::Int, 64, 1};
      case ScalarType::Half:
        return {dlpack::DataTypeCode::Float, 16, 1};
      case ScalarType::Float:
        return {dlpack::DataTypeCode::Float, 32, 1};
      case ScalarType::Double:
        return {dlpack::DataTypeCode::Float, 64, 1};
      case ScalarType::ComplexFloat:
        return {dlpack::DataTypeCode::Complex, 64, 1};
      case ScalarType::ComplexDouble:
        return {dlpack::DataTypeCode::Complex, 128, 1};
      case ScalarType::Bool:
        return {dlpack::DataTypeCode::Bool, 8, 1};
      case ScalarType::BFloat16:
        return {dlpack::DataTypeCode::BFloat, 16, 1};
      case ScalarType::UInt16:
        return {dlpack::DataTypeCode::UInt, 16, 1};
      case ScalarType::UInt32:
        return {dlpack::DataTypeCode::UInt, 32, 1};
      case ScalarType::UInt64:
        return {dlpack::DataTypeCode::UInt, 64, 1};
      default:
        throw std::runtime_error(
            "ExecuTorch result dtype is not supported by DLPack");
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
  void* data_ = nullptr;
  size_t nbytes_ = 0;
  std::vector<executorch::aten::SizesType> sizes_;
  std::vector<executorch::aten::StridesType> strides_;
  executorch::aten::ScalarType scalar_type_;
  executorch::aten::Device device_;
  std::shared_ptr<void> owner_;
};

} // namespace executorch::extension::pybindings
