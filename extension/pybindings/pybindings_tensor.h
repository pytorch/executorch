/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <executorch/runtime/core/device_allocator.h>
#include <executorch/runtime/core/device_memory_buffer.h>
#include <executorch/runtime/core/exec_aten/util/tensor_dimension_limit.h>
#include <executorch/runtime/core/portable_type/device.h>

namespace executorch {
namespace extension {
namespace pybindings {

namespace py = pybind11;

// Values match ExecuTorch and ATen ScalarType so conversion is a checked cast.
enum class PyTensorScalarType : int8_t {
  Byte = 0,
  Char = 1,
  Short = 2,
  Int = 3,
  Long = 4,
  Half = 5,
  Float = 6,
  Double = 7,
  ComplexFloat = 9,
  ComplexDouble = 10,
  Bool = 11,
  UInt16 = 27,
  UInt32 = 28,
  UInt64 = 29,
};

inline const char* device_type_name(runtime::etensor::DeviceType type) {
  switch (type) {
    case runtime::etensor::DeviceType::CPU:
      return "cpu";
    case runtime::etensor::DeviceType::CUDA:
      return "cuda";
  }
  throw std::runtime_error("Unsupported ExecuTorch device type");
}

inline std::string device_repr(runtime::etensor::Device device) {
  return "Device(type='" + std::string(device_type_name(device.type())) +
      "', index=" + std::to_string(static_cast<int>(device.index())) + ")";
}

/** A small Python-facing tensor with no ATen dependency. */
class PyTensor final {
 public:
  explicit PyTensor(
      const py::object& data,
      const py::object& dtype = py::none(),
      const py::object& dim_order = py::none(),
      runtime::etensor::Device device = runtime::etensor::DeviceType::CPU)
      : device_(device) {
    py::module_ numpy = py::module_::import("numpy");
    py::object array_source = data;
    if (PyObject_CheckBuffer(data.ptr())) {
      array_source = py::module_::import("builtins").attr("memoryview")(data);
    }
    py::object array_object = dtype.is_none()
        ? numpy.attr("asarray")(array_source)
        : numpy.attr("asarray")(array_source, py::arg("dtype") = dtype);
    py::array array = numpy.attr("ascontiguousarray")(array_object);

    scalar_type_ = scalar_type_from_numpy(array.dtype());
    sizes_.reserve(array.ndim());
    for (py::ssize_t i = 0; i < array.ndim(); ++i) {
      sizes_.push_back(checked_value<int32_t>(array.shape(i), "dimension"));
    }
    initialize_layout(dim_order);
    nbytes_ = array.nbytes();
    validate_nbytes();

    std::vector<uint8_t> storage(nbytes_);
    copy_logical_to_storage(array.data(), storage.data());
    if (device_.is_cpu()) {
      host_data_ = std::move(storage);
    } else {
      initialize_storage(storage.data());
    }
  }

  PyTensor(const PyTensor&) = delete;
  PyTensor& operator=(const PyTensor&) = delete;
  PyTensor(PyTensor&&) = delete;
  PyTensor& operator=(PyTensor&&) = delete;

  static std::shared_ptr<PyTensor> from_data(
      const void* data,
      size_t nbytes,
      const std::vector<int64_t>& sizes,
      const std::vector<int64_t>& strides,
      const std::vector<uint8_t>& dim_order,
      PyTensorScalarType scalar_type,
      runtime::etensor::Device device) {
    std::vector<int32_t> checked_sizes;
    std::vector<int32_t> checked_strides;
    checked_sizes.reserve(sizes.size());
    checked_strides.reserve(strides.size());
    for (const auto size : sizes) {
      checked_sizes.push_back(checked_value<int32_t>(size, "dimension"));
    }
    for (const auto stride : strides) {
      checked_strides.push_back(checked_value<int32_t>(stride, "stride"));
    }
    return std::shared_ptr<PyTensor>(new PyTensor(
        data,
        nbytes,
        std::move(checked_sizes),
        std::move(checked_strides),
        dim_order,
        scalar_type,
        device,
        PrivateTag{}));
  }

  const void* data() const {
    if (device_.is_cpu()) {
      return host_data_.empty() ? nullptr : host_data_.data();
    }
    return device_data_ == nullptr ? nullptr : device_data_->data();
  }

  void* mutable_data() {
    return const_cast<void*>(data());
  }

  const std::vector<int32_t>& sizes_data() const {
    return sizes_;
  }

  const std::vector<int32_t>& strides_data() const {
    return strides_;
  }

  const std::vector<uint8_t>& dim_order_data() const {
    return dim_order_;
  }

  PyTensorScalarType scalar_type() const {
    return scalar_type_;
  }

  py::buffer_info buffer() const {
    if (!device_.is_cpu()) {
      throw py::buffer_error(
          "The Python buffer protocol is only available for CPU tensors");
    }
    std::vector<py::ssize_t> shape(sizes_.begin(), sizes_.end());
    std::vector<py::ssize_t> strides;
    strides.reserve(strides_.size());
    const auto element_size = element_size_for(scalar_type_);
    for (const auto stride : strides_) {
      strides.push_back(stride * element_size);
    }
    const auto ndim = shape.size();
    return py::buffer_info(
        const_cast<void*>(data()),
        element_size,
        buffer_format(scalar_type_),
        ndim,
        std::move(shape),
        std::move(strides),
        /*readonly=*/true);
  }

  runtime::etensor::Device device_data() const {
    return device_;
  }

  py::array numpy() {
    std::vector<py::ssize_t> shape(sizes_.begin(), sizes_.end());
    std::vector<py::ssize_t> byte_strides;
    byte_strides.reserve(strides_.size());
    const auto element_size = element_size_for(scalar_type_);
    for (const auto stride : strides_) {
      byte_strides.push_back(stride * element_size);
    }

    py::array result(
        numpy_dtype(scalar_type_), std::move(shape), std::move(byte_strides));
    copy_storage_to_host(result.mutable_data());
    return result;
  }

  py::tuple sizes() const {
    py::tuple result(sizes_.size());
    for (size_t i = 0; i < sizes_.size(); ++i) {
      result[i] = sizes_[i];
    }
    return result;
  }

  py::tuple strides() const {
    py::tuple result(strides_.size());
    for (size_t i = 0; i < strides_.size(); ++i) {
      result[i] = strides_[i];
    }
    return result;
  }

  py::tuple dim_order() const {
    py::tuple result(dim_order_.size());
    for (size_t i = 0; i < dim_order_.size(); ++i) {
      result[i] = dim_order_[i];
    }
    return result;
  }

  runtime::etensor::Device device() const {
    return device_;
  }

  py::dtype dtype() const {
    return numpy_dtype(scalar_type_);
  }

  size_t nbytes() const {
    return nbytes_;
  }

  std::string repr() {
    return "Tensor(array=" + py::repr(numpy()).cast<std::string>() +
        ", dim_order=" + py::repr(dim_order()).cast<std::string>() +
        ", device=" + device_repr(device_) + ")";
  }

 private:
  struct PrivateTag final {};

  PyTensor(
      const void* data,
      size_t nbytes,
      std::vector<int32_t> sizes,
      std::vector<int32_t> strides,
      std::vector<uint8_t> dim_order,
      PyTensorScalarType scalar_type,
      runtime::etensor::Device device,
      PrivateTag)
      : nbytes_(nbytes),
        sizes_(std::move(sizes)),
        strides_(std::move(strides)),
        dim_order_(std::move(dim_order)),
        scalar_type_(scalar_type),
        device_(device) {
    if (nbytes > 0 && data == nullptr) {
      throw std::invalid_argument("Tensor data cannot be null");
    }
    validate_runtime_metadata();
    validate_nbytes();

    if (device_.is_cpu()) {
      initialize_storage(data);
    } else {
      // DeviceAllocator does not expose device-to-device copies. Stage through
      // host memory so the returned Tensor owns device storage independent of
      // the runtime output buffer and can safely be passed back as an input.
      std::vector<uint8_t> host_data(nbytes_);
      copy_device_to_host(host_data.data(), data);
      initialize_storage(host_data.data());
    }
  }

  template <typename T>
  static T checked_value(int64_t value, const char* name) {
    if (value < 0 || value > std::numeric_limits<T>::max()) {
      throw std::overflow_error(
          std::string("Tensor ") + name + " is too large for ExecuTorch");
    }
    return static_cast<T>(value);
  }

  void initialize_layout(const py::object& requested_dim_order) {
    if (sizes_.size() > runtime::kTensorDimensionLimit) {
      throw std::overflow_error("Tensor rank is too large for ExecuTorch");
    }

    if (requested_dim_order.is_none()) {
      dim_order_.resize(sizes_.size());
      for (size_t i = 0; i < dim_order_.size(); ++i) {
        dim_order_[i] = static_cast<uint8_t>(i);
      }
    } else {
      const auto order = requested_dim_order.cast<std::vector<int64_t>>();
      dim_order_.reserve(order.size());
      for (const auto dim : order) {
        dim_order_.push_back(checked_value<uint8_t>(dim, "dimension order"));
      }
    }
    validate_dim_order();
    strides_ = strides_from_dim_order();
  }

  void validate_runtime_metadata() const {
    if (sizes_.size() > runtime::kTensorDimensionLimit) {
      throw std::overflow_error("Tensor rank is too large for ExecuTorch");
    }
    if (sizes_.size() != strides_.size() ||
        sizes_.size() != dim_order_.size()) {
      throw std::invalid_argument("Tensor metadata ranks must match");
    }
    validate_dim_order();
  }

  void validate_dim_order() const {
    if (dim_order_.size() != sizes_.size()) {
      throw std::invalid_argument(
          "Tensor dimension order must have one entry per dimension");
    }
    std::vector<bool> seen(dim_order_.size(), false);
    for (const auto dim : dim_order_) {
      if (dim >= dim_order_.size() || seen[dim]) {
        throw std::invalid_argument(
            "Tensor dimension order must be a permutation of its dimensions");
      }
      seen[dim] = true;
    }
  }

  std::vector<int32_t> strides_from_dim_order() const {
    std::vector<int32_t> result(sizes_.size());
    if (sizes_.empty()) {
      return result;
    }

    int32_t stride = 1;
    for (size_t i = dim_order_.size(); i > 0; --i) {
      const auto dim = dim_order_[i - 1];
      result[dim] = stride;
      const auto size = sizes_[dim];
      const auto stride_size = size == 0 ? 1 : size;
      if (stride > std::numeric_limits<int32_t>::max() / stride_size) {
        throw std::overflow_error("Tensor stride is too large for ExecuTorch");
      }
      stride *= stride_size;
    }
    return result;
  }

  void copy_logical_to_storage(const void* source, void* destination) const {
    if (nbytes_ == 0) {
      return;
    }
    bool is_contiguous = true;
    for (size_t i = 0; i < dim_order_.size(); ++i) {
      if (dim_order_[i] != i) {
        is_contiguous = false;
        break;
      }
    }
    if (is_contiguous) {
      std::memcpy(destination, source, nbytes_);
      return;
    }

    const auto element_size = element_size_for(scalar_type_);
    const auto numel = nbytes_ / element_size;
    const auto* source_bytes = static_cast<const uint8_t*>(source);
    auto* destination_bytes = static_cast<uint8_t*>(destination);
    std::vector<int32_t> coordinates(sizes_.size(), 0);
    size_t storage_offset = 0;
    for (size_t logical_offset = 0; logical_offset < numel; ++logical_offset) {
      std::memcpy(
          destination_bytes + storage_offset * element_size,
          source_bytes + logical_offset * element_size,
          element_size);

      for (size_t i = sizes_.size(); i > 0; --i) {
        const auto dim = i - 1;
        ++coordinates[dim];
        storage_offset += strides_[dim];
        if (coordinates[dim] < sizes_[dim]) {
          break;
        }
        storage_offset -= coordinates[dim] * strides_[dim];
        coordinates[dim] = 0;
      }
    }
  }

  void initialize_storage(const void* host_data) {
    if (nbytes_ == 0) {
      return;
    }
    if (device_.is_cpu()) {
      host_data_.resize(nbytes_);
      std::memcpy(host_data_.data(), host_data, nbytes_);
      return;
    }

    auto* allocator = runtime::get_device_allocator(device_.type());
    if (allocator == nullptr) {
      throw std::runtime_error(
          "No allocator is registered for the lightweight Tensor device");
    }
    auto buffer = runtime::DeviceMemoryBuffer::create(
        nbytes_, device_.type(), device_.index());
    if (!buffer.ok()) {
      throw std::runtime_error(
          "Failed to allocate lightweight Tensor device storage (error " +
          std::to_string(static_cast<uint32_t>(buffer.error())) + ")");
    }
    auto owned_buffer =
        std::make_unique<runtime::DeviceMemoryBuffer>(std::move(buffer.get()));
    const auto error = allocator->copy_host_to_device(
        owned_buffer->data(), host_data, nbytes_, device_.index());
    if (error != runtime::Error::Ok) {
      throw std::runtime_error(
          "Failed to copy lightweight Tensor data to its device (error " +
          std::to_string(static_cast<uint32_t>(error)) + ")");
    }
    device_data_ = std::move(owned_buffer);
  }

  void copy_device_to_host(void* destination, const void* source) const {
    if (nbytes_ == 0) {
      return;
    }
    auto* allocator = runtime::get_device_allocator(device_.type());
    if (allocator == nullptr) {
      throw std::runtime_error(
          "No allocator is registered for the lightweight Tensor device");
    }
    const auto error = allocator->copy_device_to_host(
        destination, source, nbytes_, device_.index());
    if (error != runtime::Error::Ok) {
      throw std::runtime_error(
          "Failed to copy lightweight Tensor data from its device (error " +
          std::to_string(static_cast<uint32_t>(error)) + ")");
    }
  }

  void copy_storage_to_host(void* destination) const {
    if (device_.is_cpu()) {
      if (nbytes_ > 0) {
        std::memcpy(destination, host_data_.data(), nbytes_);
      }
      return;
    }
    copy_device_to_host(destination, data());
  }

  void validate_nbytes() const {
    size_t expected = element_size_for(scalar_type_);
    for (const auto size : sizes_) {
      if (size == 0) {
        expected = 0;
        break;
      }
      if (expected > std::numeric_limits<size_t>::max() / size) {
        throw std::overflow_error("Tensor byte size overflow");
      }
      expected *= size;
    }
    if (nbytes_ != expected) {
      throw std::invalid_argument("Tensor data size does not match its shape");
    }
  }

  static PyTensorScalarType scalar_type_from_numpy(const py::dtype& dtype) {
    if (!dtype.attr("isnative").cast<bool>()) {
      throw std::invalid_argument(
          "Tensor only supports NumPy arrays with native byte order");
    }

    const std::string name = py::str(dtype.attr("name"));
    if (name == "uint8") {
      return PyTensorScalarType::Byte;
    }
    if (name == "int8") {
      return PyTensorScalarType::Char;
    }
    if (name == "int16") {
      return PyTensorScalarType::Short;
    }
    if (name == "int32") {
      return PyTensorScalarType::Int;
    }
    if (name == "int64") {
      return PyTensorScalarType::Long;
    }
    if (name == "float16") {
      return PyTensorScalarType::Half;
    }
    if (name == "float32") {
      return PyTensorScalarType::Float;
    }
    if (name == "float64") {
      return PyTensorScalarType::Double;
    }
    if (name == "complex64") {
      return PyTensorScalarType::ComplexFloat;
    }
    if (name == "complex128") {
      return PyTensorScalarType::ComplexDouble;
    }
    if (name == "bool") {
      return PyTensorScalarType::Bool;
    }
    if (name == "uint16") {
      return PyTensorScalarType::UInt16;
    }
    if (name == "uint32") {
      return PyTensorScalarType::UInt32;
    }
    if (name == "uint64") {
      return PyTensorScalarType::UInt64;
    }
    throw std::invalid_argument("Unsupported NumPy dtype: " + name);
  }

  static py::dtype numpy_dtype(PyTensorScalarType scalar_type) {
    switch (scalar_type) {
      case PyTensorScalarType::Byte:
        return py::dtype("uint8");
      case PyTensorScalarType::Char:
        return py::dtype("int8");
      case PyTensorScalarType::Short:
        return py::dtype("int16");
      case PyTensorScalarType::Int:
        return py::dtype("int32");
      case PyTensorScalarType::Long:
        return py::dtype("int64");
      case PyTensorScalarType::Half:
        return py::dtype("float16");
      case PyTensorScalarType::Float:
        return py::dtype("float32");
      case PyTensorScalarType::Double:
        return py::dtype("float64");
      case PyTensorScalarType::ComplexFloat:
        return py::dtype("complex64");
      case PyTensorScalarType::ComplexDouble:
        return py::dtype("complex128");
      case PyTensorScalarType::Bool:
        return py::dtype("bool");
      case PyTensorScalarType::UInt16:
        return py::dtype("uint16");
      case PyTensorScalarType::UInt32:
        return py::dtype("uint32");
      case PyTensorScalarType::UInt64:
        return py::dtype("uint64");
    }
    throw std::runtime_error("Unsupported ExecuTorch Tensor dtype");
  }

  static const char* buffer_format(PyTensorScalarType scalar_type) {
    switch (scalar_type) {
      case PyTensorScalarType::Byte:
        return "B";
      case PyTensorScalarType::Char:
        return "b";
      case PyTensorScalarType::Short:
        return "h";
      case PyTensorScalarType::Int:
        return "i";
      case PyTensorScalarType::Long:
        return "q";
      case PyTensorScalarType::Half:
        return "e";
      case PyTensorScalarType::Float:
        return "f";
      case PyTensorScalarType::Double:
        return "d";
      case PyTensorScalarType::ComplexFloat:
        return "Zf";
      case PyTensorScalarType::ComplexDouble:
        return "Zd";
      case PyTensorScalarType::Bool:
        return "?";
      case PyTensorScalarType::UInt16:
        return "H";
      case PyTensorScalarType::UInt32:
        return "I";
      case PyTensorScalarType::UInt64:
        return "Q";
    }
    throw std::runtime_error("Unsupported ExecuTorch Tensor dtype");
  }

  static size_t element_size_for(PyTensorScalarType scalar_type) {
    switch (scalar_type) {
      case PyTensorScalarType::Byte:
      case PyTensorScalarType::Char:
      case PyTensorScalarType::Bool:
        return 1;
      case PyTensorScalarType::Short:
      case PyTensorScalarType::Half:
      case PyTensorScalarType::UInt16:
        return 2;
      case PyTensorScalarType::Int:
      case PyTensorScalarType::Float:
      case PyTensorScalarType::UInt32:
        return 4;
      case PyTensorScalarType::Long:
      case PyTensorScalarType::Double:
      case PyTensorScalarType::ComplexFloat:
      case PyTensorScalarType::UInt64:
        return 8;
      case PyTensorScalarType::ComplexDouble:
        return 16;
    }
    throw std::runtime_error("Unsupported ExecuTorch Tensor dtype");
  }

  std::vector<uint8_t> host_data_;
  std::unique_ptr<runtime::DeviceMemoryBuffer> device_data_;
  size_t nbytes_ = 0;
  std::vector<int32_t> sizes_;
  std::vector<int32_t> strides_;
  std::vector<uint8_t> dim_order_;
  PyTensorScalarType scalar_type_ = PyTensorScalarType::Float;
  runtime::etensor::Device device_ = runtime::etensor::DeviceType::CPU;
};

} // namespace pybindings
} // namespace extension
} // namespace executorch
