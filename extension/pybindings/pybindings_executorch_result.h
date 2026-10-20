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

#include <executorch/extension/pybindings/pybindings_cuda.h>
#include <executorch/extension/pybindings/pybindings_dlpack.h>
#include <executorch/runtime/core/device_memory_buffer.h>
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

inline PyObject* loaded_torch_module() {
  // Retain the module for the interpreter's lifetime. This avoids repeated
  // sys.modules lookups without creating a static py::object whose destructor
  // could run after Python has finalized.
  static PyObject* torch_module = nullptr;
  if (torch_module == nullptr) {
    PyObject* loaded = PyDict_GetItemString(PyImport_GetModuleDict(), "torch");
    if (loaded != nullptr) {
      Py_INCREF(loaded);
      torch_module = loaded;
    }
  }
  return torch_module;
}

inline void synchronize_torch_cuda_device(int index) {
  py::reinterpret_borrow<py::object>(loaded_torch_module())
      .attr("cuda")
      .attr("synchronize")(index);
}

// Waits for every queued operation on a device. Bindings built without CUDA
// support have no CUDA runtime of their own, so they wait through torch's,
// which is loaded whenever such bindings see CUDA memory.
inline void synchronize_device(int index) {
#ifndef EXECUTORCH_PYBINDINGS_CUDA
  if (loaded_torch_module() != nullptr) {
    synchronize_torch_cuda_device(index);
    return;
  }
#endif
  synchronize_cuda_device(index);
}

/** Tensor result returned by portable Python bindings. */
class PyExecuTorchResult final
    : public std::enable_shared_from_this<PyExecuTorchResult> {
 public:
  explicit PyExecuTorchResult(
      const executorch::aten::Tensor& tensor,
      bool clone = true,
      py::object owner = py::none())
      : sizes_(tensor.sizes().begin(), tensor.sizes().end()),
        strides_(tensor.strides().begin(), tensor.strides().end()),
        scalar_type_(tensor.scalar_type()),
        nbytes_(tensor.nbytes()),
        device_(tensor.device()),
        owner_(std::move(owner)) {
    if (!device_.is_cpu() &&
        device_.type() != executorch::aten::DeviceType::CUDA) {
      throw std::runtime_error(
          "ExecuTorchResult supports only CPU and CUDA outputs");
    }
    if (nbytes_ != 0 && tensor.const_data_ptr() == nullptr) {
      throw py::value_error("ExecuTorch result data is not allocated");
    }
    if (clone && device_.is_cpu()) {
      storage_.resize(nbytes_);
      if (!storage_.empty()) {
        std::memcpy(storage_.data(), tensor.const_data_ptr(), storage_.size());
      }
      data_ = storage_.data();
      owner_ = py::none();
    } else if (clone) {
      if (nbytes_ != 0) {
        device_storage_ = allocate_cuda_result(nbytes_, device_.index());
        copy_cuda_storage(
            device_storage_.get(),
            tensor.const_data_ptr(),
            nbytes_,
            device_.index());
        // The source can be memory the caller owns, such as an input the
        // method returns, and the caller may write it on another stream as
        // soon as this call returns. A clone must not see that write.
        synchronize_cuda_device(device_.index());
        data_ = device_storage_.get();
      }
      owner_ = py::none();
    } else {
      data_ = const_cast<void*>(tensor.const_data_ptr());
    }
  }

  py::buffer_info buffer() {
    require_cpu_buffer();
    // PEP 3118 has no bfloat16 format. Expose its physical bytes without
    // advertising a different numeric dtype; DLPack preserves the real dtype.
    if (scalar_type_ == executorch::aten::ScalarType::BFloat16) {
      return flat_buffer();
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
        require_buffer_format(),
        ndim,
        std::move(shape),
        std::move(byte_strides),
        /*readonly=*/false);
  }

  py::buffer_info flat_buffer() {
    require_cpu_buffer();
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
          " cannot be represented by NumPy; dtype_name names it");
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

  py::object backing_owner(const py::object& self) const {
    return owner_.is_none() ? self : owner_;
  }

  static py::object dlpack_backing_owner(dlpack::ManagedTensor* managed) {
    if (managed == nullptr || managed->deleter != release_dlpack_holder ||
        managed->manager_ctx == nullptr) {
      return py::none();
    }
    const auto* holder = static_cast<const DLPackHolder*>(managed->manager_ctx);
    if (&holder->managed != managed ||
        managed->dl_tensor.data != holder->result->data_ ||
        managed->dl_tensor.byte_offset != 0) {
      return py::none();
    }
    const auto& result = holder->result;
    return result->owner_.is_none() ? py::cast(result) : result->owner_;
  }

  py::tuple dlpack_device() const {
    const auto device = dl_device();
    return py::make_tuple(
        static_cast<int>(device.device_type), device.device_id);
  }

  py::capsule to_dlpack(const py::object& stream = py::none()) {
    // Always wait for the device before handing CUDA memory to another
    // library: the result may share memory a call still has work queued on,
    // or an upload that is still running.
    if (!device_.is_cpu() && nbytes_ != 0) {
      synchronize_device(device_.index());
    }
    (void)stream;
    return capsule();
  }

  // A capsule for these bindings to read the result as a method input. It
  // skips the wait in to_dlpack(), since the call waits for every CUDA input
  // before it runs.
  py::capsule capsule() {
    auto holder = std::make_unique<DLPackHolder>(shared_from_this());
    py::capsule capsule(&holder->managed, "dltensor", [](PyObject* capsule) {
      if (PyCapsule_IsValid(capsule, "dltensor")) {
        auto* managed = static_cast<dlpack::ManagedTensor*>(
            PyCapsule_GetPointer(capsule, "dltensor"));
        managed->deleter(managed);
      }
    });
    holder.release();
    return capsule;
  }

  const char* torch_dtype_name() const {
    return python_dtype(scalar_type_).torch_name;
  }

  // Lets a caller without torch read a CUDA result, or keep a CPU copy that
  // later calls cannot overwrite.
  std::shared_ptr<PyExecuTorchResult> to_host() const {
    std::vector<uint8_t> storage(nbytes_);
    if (nbytes_ != 0) {
      if (device_.is_cpu()) {
        std::memcpy(storage.data(), data_, nbytes_);
      } else {
        // Wait before the copy: the result may share memory a call still has
        // work queued on, or an upload that is still running.
        synchronize_cuda_device(device_.index());
        copy_cuda_storage(
            storage.data(),
            data_,
            nbytes_,
            device_.index(),
            CudaCopy::DeviceToHost);
      }
    }
    auto result = std::shared_ptr<PyExecuTorchResult>(new PyExecuTorchResult(
        sizes_,
        strides_,
        scalar_type_,
        executorch::aten::Device(executorch::aten::DeviceType::CPU, 0)));
    result->storage_ = std::move(storage);
    result->data_ = result->storage_.data();
    return result;
  }

  // Copies a CPU result to a CUDA device, keeping its dtype and layout, which
  // a buffer export cannot carry for dtypes such as bfloat16.
  std::shared_ptr<PyExecuTorchResult> to_device(int device_index) const {
    if (!device_.is_cpu()) {
      throw py::value_error(
          "from_host() takes a CPU result; this one is already on a device");
    }
    return from_host(data_, sizes_, strides_, scalar_type_, device_index);
  }

  // Lets a caller without torch place host data on a CUDA device, to pass as
  // an input to a method whose inputs live there. The data must be dense in
  // the given strides, as for any ExecuTorch input.
  static std::shared_ptr<PyExecuTorchResult> from_host(
      const void* data,
      std::vector<executorch::aten::SizesType> sizes,
      std::vector<executorch::aten::StridesType> strides,
      executorch::aten::ScalarType scalar_type,
      int device_index) {
    if (device_index < 0 ||
        device_index > std::numeric_limits<
                           executorch::runtime::etensor::DeviceIndex>::max()) {
      throw py::value_error(
          "CUDA device index " + std::to_string(device_index) +
          " is out of range");
    }
    require_cuda_device(device_index);
    auto result = std::shared_ptr<PyExecuTorchResult>(new PyExecuTorchResult(
        std::move(sizes),
        std::move(strides),
        scalar_type,
        executorch::aten::Device(
            executorch::aten::DeviceType::CUDA,
            static_cast<executorch::runtime::etensor::DeviceIndex>(
                device_index))));
    if (result->nbytes_ != 0) {
      result->device_storage_ =
          allocate_cuda_result(result->nbytes_, device_index);
      copy_cuda_storage(
          result->device_storage_.get(),
          data,
          result->nbytes_,
          device_index,
          CudaCopy::HostToDevice);
      result->data_ = result->device_storage_.get();
    }
    return result;
  }

  // `storage` must own the memory `tensor` points at; the result reads that
  // memory and keeps `storage` alive.
  static std::shared_ptr<PyExecuTorchResult> hand_over(
      const executorch::aten::Tensor& tensor,
      std::shared_ptr<void> storage) {
    auto result = std::make_shared<PyExecuTorchResult>(tensor, /*clone=*/false);
    result->device_storage_ = std::move(storage);
    return result;
  }

  std::string repr() const {
    std::string shape;
    for (size_t i = 0; i < sizes_.size(); ++i) {
      shape += (i == 0 ? "" : ", ") + std::to_string(sizes_[i]);
    }
    if (sizes_.size() == 1) {
      shape += ",";
    }
    const std::string device =
        device_.is_cpu() ? "cpu" : "cuda:" + std::to_string(device_.index());
    return "ExecuTorchResult(shape=(" + shape +
        "), dtype=" + torch_dtype_name() + ", device=" + device + ")";
  }

 private:
  // A result with this layout and no data yet, for the factories above.
  PyExecuTorchResult(
      std::vector<executorch::aten::SizesType> sizes,
      std::vector<executorch::aten::StridesType> strides,
      executorch::aten::ScalarType scalar_type,
      executorch::aten::Device device)
      : sizes_(std::move(sizes)),
        strides_(std::move(strides)),
        scalar_type_(scalar_type),
        nbytes_(
            static_cast<size_t>(executorch::aten::compute_numel(
                sizes_.data(),
                static_cast<ssize_t>(sizes_.size()))) *
            executorch::runtime::elementSize(scalar_type)),
        device_(device),
        owner_(py::none()) {}

  struct DLPackHolder final {
    explicit DLPackHolder(std::shared_ptr<PyExecuTorchResult> value)
        : result(std::move(value)),
          shape(result->sizes_.begin(), result->sizes_.end()),
          strides(result->strides_.begin(), result->strides_.end()) {
      managed.dl_tensor = {
          result->data_,
          result->dl_device(),
          static_cast<int32_t>(shape.size()),
          result->dl_dtype(),
          shape.data(),
          strides.data(),
          0};
      managed.manager_ctx = this;
      managed.deleter = release_dlpack_holder;
    }

    std::shared_ptr<PyExecuTorchResult> result;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    dlpack::ManagedTensor managed{};
  };

  static void release_dlpack_holder(dlpack::ManagedTensor* managed) {
    // A native DLPack consumer can release its tensor on a worker thread.
    // The retained result may own Python buffer exports and capsules.
    py::gil_scoped_acquire gil;
    delete static_cast<DLPackHolder*>(managed->manager_ctx);
  }

  dlpack::Device dl_device() const {
    return {
        device_.is_cpu() ? dlpack::DeviceType::CPU : dlpack::DeviceType::CUDA,
        device_.is_cpu() ? 0 : device_.index()};
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
      case ScalarType::ComplexHalf:
        return {dlpack::DataTypeCode::Complex, 32, 1};
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
        throw py::value_error(
            "ExecuTorch result dtype cannot be represented by DLPack");
    }
  }

  void require_cpu_buffer() const {
    if (!device_.is_cpu()) {
      throw py::buffer_error(
          "CUDA ExecuTorch results use DLPack instead of the Python buffer protocol");
    }
  }

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
  std::shared_ptr<void> device_storage_;
  std::vector<executorch::aten::SizesType> sizes_;
  std::vector<executorch::aten::StridesType> strides_;
  executorch::aten::ScalarType scalar_type_;
  size_t nbytes_;
  executorch::aten::Device device_;
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
