/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <executorch/devtools/bundled_program/bundled_program.h>
#include <executorch/devtools/bundled_program/schema/bundled_program_schema_generated.h>
#include <executorch/devtools/etdump/etdump_flatcc.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/data_loader/mmap_data_loader.h>
#include <executorch/extension/flat_tensor/flat_tensor_data_map.h>
#include <executorch/extension/memory_allocator/malloc_memory_allocator.h>
#include <executorch/extension/module/bundled_module.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/pybindings/pybindings_data_loader.h>
#include <executorch/extension/pybindings/pybindings_executorch_result.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/extension/threadpool/threadpool.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/device_memory_buffer.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_dimension_limit.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>

#ifdef USE_ATEN_LIB
#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/core/functional.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>
#endif

/// Throws a runtime_error with the provided message if `error` is not `Ok`.
#define THROW_IF_ERROR(error, message, ...)                       \
  ({                                                              \
    if ((error) != Error::Ok) {                                   \
      char msg_buf[128];                                          \
      snprintf(msg_buf, sizeof(msg_buf), message, ##__VA_ARGS__); \
      /* pybind will convert this to a python exception. */       \
      throw std::runtime_error(msg_buf);                          \
    }                                                             \
  })

#define THROW_INDEX_IF_ERROR(error, message, ...)                 \
  ({                                                              \
    if ((error) != Error::Ok) {                                   \
      char msg_buf[128];                                          \
      snprintf(msg_buf, sizeof(msg_buf), message, ##__VA_ARGS__); \
      /* pybind will convert this to a python exception. */       \
      throw std::out_of_range(msg_buf);                           \
    }                                                             \
  })

namespace py = pybind11;
using ::executorch::ET_RUNTIME_NAMESPACE::BackendInterface;
using ::executorch::ET_RUNTIME_NAMESPACE::get_backend_class;
using ::executorch::ET_RUNTIME_NAMESPACE::get_backend_name;
using ::executorch::ET_RUNTIME_NAMESPACE::get_num_registered_backends;
using ::executorch::ET_RUNTIME_NAMESPACE::get_registered_kernels;
using ::executorch::ET_RUNTIME_NAMESPACE::Kernel;
using ::executorch::ET_RUNTIME_NAMESPACE::Method;
using ::executorch::ET_RUNTIME_NAMESPACE::MethodMeta;
using ::executorch::ET_RUNTIME_NAMESPACE::Program;
using ::executorch::extension::BufferDataLoader;
using ::executorch::extension::MallocMemoryAllocator;
using ::executorch::extension::MmapDataLoader;
using ::executorch::extension::ET_BUNDLED_MODULE_NAMESPACE::BundledModule;
using ::executorch::extension::pybindings::PyDataLoader;
using ::executorch::runtime::DataLoader;
using ::executorch::runtime::DeviceMemoryBuffer;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::EventTracerDebugLogLevel;
using ::executorch::runtime::HierarchicalAllocator;
using ::executorch::runtime::MemoryAllocator;
using ::executorch::runtime::MemoryManager;
using ::executorch::runtime::prof_result_t;
using ::executorch::runtime::Result;
using ::executorch::runtime::Span;
using ::executorch::runtime::Tag;
using torch::executor::etdump_result;
using torch::executor::ETDumpGen;

namespace executorch {
namespace extension {
namespace pybindings {

namespace {

#ifndef USE_ATEN_LIB
constexpr char kTensorOutputEnvironmentVariable[] =
    "EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT";

enum class PortableTensorOutput { Torch, ExecuTorch };

PortableTensorOutput& portable_tensor_output() {
  static PortableTensorOutput output = PortableTensorOutput::Torch;
  return output;
}

void configure_portable_tensor_output() {
  const char* value = std::getenv(kTensorOutputEnvironmentVariable);
  if (value == nullptr || std::strcmp(value, "torch") == 0) {
    portable_tensor_output() = PortableTensorOutput::Torch;
    return;
  }
  if (std::strcmp(value, "executorch") == 0) {
    portable_tensor_output() = PortableTensorOutput::ExecuTorch;
    return;
  }
  throw py::value_error(
      std::string(kTensorOutputEnvironmentVariable) +
      " must be either 'torch' or 'executorch', but was '" + value + "'");
}
#endif

#ifdef USE_ATEN_LIB
executorch::aten::ScalarType runtime_scalar_type(const at::Tensor& tensor) {
  return tensor.scalar_type();
}

void* mutable_tensor_data_ptr_no_cow(at::Tensor& tensor) {
  if (tensor.numel() == 0) {
    return nullptr;
  }

  if (!tensor.has_storage()) {
    throw py::value_error(
        "Tensor has a non-zero number of elements, but its data is not allocated");
  }

  auto* storage_data = static_cast<char*>(tensor.unsafeGetTensorImpl()
                                              ->unsafe_storage()
                                              .unsafeGetStorageImpl()
                                              ->_mutable_data_ptr_no_checks()
                                              .mutable_get());
  if (storage_data == nullptr) {
    throw py::value_error(
        "Tensor has a non-zero number of elements, but its data is not allocated");
  }
  return storage_data + tensor.storage_offset() * tensor.itemsize();
}
#endif

/** Holds a Python buffer export and presents its metadata to ExecuTorch. */
class BufferTensor final {
 public:
  explicit BufferTensor(const py::handle& value)
      : buffer_(py::reinterpret_borrow<py::buffer>(value)),
        info_(buffer_.request()) {
    if (info_.ndim < 0 ||
        static_cast<size_t>(info_.ndim) > runtime::kTensorDimensionLimit) {
      throw py::buffer_error("Buffer rank is too large for ExecuTorch");
    }
    scalar_type_ = scalar_type_from_buffer(info_);
    sizes_.reserve(info_.shape.size());
    strides_.reserve(info_.strides.size());
    for (const auto size : info_.shape) {
      sizes_.push_back(checked_int(size, "dimension"));
    }
    for (const auto byte_stride : info_.strides) {
      if (byte_stride < 0 || byte_stride % info_.itemsize != 0) {
        throw py::buffer_error(
            "ExecuTorch inputs require non-negative, element-aligned strides");
      }
      strides_.push_back(checked_int(byte_stride / info_.itemsize, "stride"));
    }
    dim_order_.resize(sizes_.size());
    std::iota(dim_order_.begin(), dim_order_.end(), 0);
    std::stable_sort(
        dim_order_.begin(), dim_order_.end(), [this](uint8_t a, uint8_t b) {
          return strides_[a] > strides_[b];
        });
    validate_dense_layout();
    if (info_.readonly && info_.size > 0) {
      owned_data_.resize(info_.size * info_.itemsize);
      std::memcpy(owned_data_.data(), info_.ptr, owned_data_.size());
    }
  }

  void* data() const {
    return owned_data_.empty() ? info_.ptr
                               : const_cast<uint8_t*>(owned_data_.data());
  }

  const std::vector<int>& sizes() const {
    return sizes_;
  }

  const std::vector<int>& strides() const {
    return strides_;
  }

  const std::vector<uint8_t>& dim_order() const {
    return dim_order_;
  }

  executorch::aten::ScalarType scalar_type() const {
    return scalar_type_;
  }

 private:
  static int checked_int(py::ssize_t value, const char* name) {
    if (value < 0 || value > std::numeric_limits<int>::max()) {
      throw py::buffer_error(
          std::string("Buffer ") + name + " is too large for ExecuTorch");
    }
    return static_cast<int>(value);
  }

  template <typename T>
  static bool has_format(const py::buffer_info& info) {
    return info.item_type_is_equivalent_to<T>();
  }

  static executorch::aten::ScalarType scalar_type_from_buffer(
      const py::buffer_info& info) {
    if (has_format<uint8_t>(info)) {
      return executorch::aten::ScalarType::Byte;
    }
    if (has_format<int8_t>(info)) {
      return executorch::aten::ScalarType::Char;
    }
    if (has_format<int16_t>(info)) {
      return executorch::aten::ScalarType::Short;
    }
    if (has_format<int32_t>(info)) {
      return executorch::aten::ScalarType::Int;
    }
    if (has_format<int64_t>(info)) {
      return executorch::aten::ScalarType::Long;
    }
    if (info.itemsize == 2 && info.format == "e") {
      return executorch::aten::ScalarType::Half;
    }
    if (has_format<float>(info)) {
      return executorch::aten::ScalarType::Float;
    }
    if (has_format<double>(info)) {
      return executorch::aten::ScalarType::Double;
    }
    if (has_format<bool>(info)) {
      return executorch::aten::ScalarType::Bool;
    }
    if (has_format<uint16_t>(info)) {
      return executorch::aten::ScalarType::UInt16;
    }
    if (has_format<uint32_t>(info)) {
      return executorch::aten::ScalarType::UInt32;
    }
    if (has_format<uint64_t>(info)) {
      return executorch::aten::ScalarType::UInt64;
    }
    throw py::buffer_error("Unsupported buffer format: " + info.format);
  }

  void validate_dense_layout() const {
    int64_t expected_stride = 1;
    for (size_t i = dim_order_.size(); i > 0; --i) {
      const auto dim = dim_order_[i - 1];
      const auto size = sizes_[dim];
      if (size > 1 && strides_[dim] != expected_stride) {
        throw py::buffer_error(
            "ExecuTorch inputs must have a dense, non-overlapping layout");
      }
      expected_stride *= std::max(size, 1);
      if (expected_stride > std::numeric_limits<int>::max()) {
        throw py::buffer_error("Buffer is too large for ExecuTorch");
      }
    }
  }

  py::buffer buffer_;
  py::buffer_info info_;
  std::vector<uint8_t> owned_data_;
  std::vector<int> sizes_;
  std::vector<int> strides_;
  std::vector<uint8_t> dim_order_;
  executorch::aten::ScalarType scalar_type_;
};

bool is_torch_tensor(const py::handle& value) {
  const auto modules = py::module_::import("sys").attr("modules");
  if (!modules.contains("torch")) {
    return false;
  }
  return py::isinstance(value, modules["torch"].attr("Tensor"));
}

void validate_tensor_input(
    const MethodMeta& method_meta,
    size_t index,
    executorch::aten::ScalarType scalar_type,
    const std::vector<int>& sizes,
    const std::vector<int>& strides) {
  const auto input_meta = method_meta.input_tensor_meta(index);
  THROW_IF_ERROR(input_meta.error(), "Input %zu is not a tensor input", index);
  if (input_meta->scalar_type() != scalar_type) {
    throw py::value_error(
        "Input " + std::to_string(index) + " has dtype " +
        std::string(runtime::toString(scalar_type)) +
        " but the method expects " +
        std::string(runtime::toString(input_meta->scalar_type())));
  }
  const auto expected_order = input_meta->dim_order();
  if (expected_order.size() != sizes.size()) {
    throw py::value_error(
        "Input " + std::to_string(index) + " has rank " +
        std::to_string(sizes.size()) + ", but the method expects rank " +
        std::to_string(expected_order.size()));
  }
  if (strides.size() != sizes.size()) {
    throw py::value_error(
        "Input " + std::to_string(index) +
        " has inconsistent shape and stride ranks");
  }
  std::vector<int> expected_strides(sizes.size());
  const auto status = runtime::dim_order_to_stride(
      sizes.data(),
      expected_order.data(),
      sizes.size(),
      expected_strides.data());
  THROW_IF_ERROR(status, "Invalid dimension order for input %zu", index);
  for (size_t dim = 0; dim < sizes.size(); ++dim) {
    if (sizes[dim] > 1 && strides[dim] != expected_strides[dim]) {
      throw py::value_error(
          "Input " + std::to_string(index) + " has stride " +
          std::to_string(strides[dim]) + " at dimension " +
          std::to_string(dim) + ", but the method expects stride " +
          std::to_string(expected_strides[dim]));
    }
  }
}

#ifndef USE_ATEN_LIB
/** A zero-copy view built only from torch.Tensor's public Python API. */
class TorchTensorView final {
 public:
  explicit TorchTensorView(const py::handle& value)
      : owner_(py::reinterpret_borrow<py::object>(value)) {
    if (owner_.attr("is_neg")().cast<bool>() ||
        owner_.attr("is_conj")().cast<bool>()) {
      throw py::value_error(
          "Lazy conjugate and negative torch tensor views must be resolved before execution");
    }
    const auto device_type = py::str(owner_.attr("device").attr("type"));
    if (device_type.cast<std::string>() != "cpu") {
      throw py::value_error(
          "Portable bindings require CPU torch tensors until DLPack is enabled");
    }
    scalar_type_ = scalar_type_from_torch(py::str(owner_.attr("dtype")));
    sizes_ = checked_vector(owner_.attr("shape"), "dimension");
    strides_ = checked_vector(owner_.attr("stride")(), "stride");
    if (sizes_.size() > runtime::kTensorDimensionLimit ||
        sizes_.size() != strides_.size()) {
      throw py::value_error("Invalid torch tensor metadata");
    }
    dim_order_.resize(sizes_.size());
    std::iota(dim_order_.begin(), dim_order_.end(), 0);
    std::stable_sort(
        dim_order_.begin(), dim_order_.end(), [this](uint8_t a, uint8_t b) {
          return strides_[a] > strides_[b];
        });
    validate_dense_layout();
    const auto address = owner_.attr("data_ptr")().cast<uintptr_t>();
    if (address == 0 && owner_.attr("numel")().cast<int64_t>() != 0) {
      throw py::value_error("Torch tensor data is not allocated");
    }
    data_ = reinterpret_cast<void*>(address);
  }

  void* data() const {
    return data_;
  }

  const std::vector<int>& sizes() const {
    return sizes_;
  }

  const std::vector<int>& strides() const {
    return strides_;
  }

  const std::vector<uint8_t>& dim_order() const {
    return dim_order_;
  }

  executorch::aten::ScalarType scalar_type() const {
    return scalar_type_;
  }

 private:
  static std::vector<int> checked_vector(
      const py::handle& values,
      const char* name) {
    std::vector<int> result;
    for (const auto value : py::reinterpret_borrow<py::iterable>(values)) {
      const auto number = py::cast<int64_t>(value);
      if (number < 0 || number > std::numeric_limits<int>::max()) {
        throw py::value_error(
            std::string("Torch tensor ") + name +
            " is too large for ExecuTorch");
      }
      result.push_back(static_cast<int>(number));
    }
    return result;
  }

  static executorch::aten::ScalarType scalar_type_from_torch(
      const py::str& dtype) {
    const auto name = dtype.cast<std::string>();
    using executorch::aten::ScalarType;
    if (name == "torch.uint8") {
      return ScalarType::Byte;
    }
    if (name == "torch.int8") {
      return ScalarType::Char;
    }
    if (name == "torch.int16") {
      return ScalarType::Short;
    }
    if (name == "torch.int32") {
      return ScalarType::Int;
    }
    if (name == "torch.int64") {
      return ScalarType::Long;
    }
    if (name == "torch.float16") {
      return ScalarType::Half;
    }
    if (name == "torch.float32") {
      return ScalarType::Float;
    }
    if (name == "torch.float64") {
      return ScalarType::Double;
    }
    if (name == "torch.complex64") {
      return ScalarType::ComplexFloat;
    }
    if (name == "torch.complex128") {
      return ScalarType::ComplexDouble;
    }
    if (name == "torch.bool") {
      return ScalarType::Bool;
    }
    if (name == "torch.bfloat16") {
      return ScalarType::BFloat16;
    }
    if (name == "torch.uint16") {
      return ScalarType::UInt16;
    }
    if (name == "torch.uint32") {
      return ScalarType::UInt32;
    }
    if (name == "torch.uint64") {
      return ScalarType::UInt64;
    }
    throw py::value_error("Unsupported torch dtype: " + name);
  }

  void validate_dense_layout() const {
    int64_t expected_stride = 1;
    for (size_t i = dim_order_.size(); i > 0; --i) {
      const auto dim = dim_order_[i - 1];
      const auto size = sizes_[dim];
      if (size > 1 && strides_[dim] != expected_stride) {
        throw py::value_error(
            "ExecuTorch inputs must have a dense, non-overlapping layout");
      }
      expected_stride *= std::max(size, 1);
    }
  }

  py::object owner_;
  void* data_ = nullptr;
  std::vector<int> sizes_;
  std::vector<int> strides_;
  std::vector<uint8_t> dim_order_;
  executorch::aten::ScalarType scalar_type_;
};
#endif

py::sequence normalize_inputs(const py::object& inputs) {
  if (PyObject_CheckBuffer(inputs.ptr()) || is_torch_tensor(inputs)) {
    py::list result;
    result.append(inputs);
    return result;
  }
  if (!py::isinstance<py::sequence>(inputs)) {
    throw py::type_error("Inputs must be a tensor buffer or a flat sequence");
  }
  return py::reinterpret_borrow<py::sequence>(inputs);
}

#ifndef USE_ATEN_LIB
py::object portable_tensor_result(const executorch::aten::Tensor& tensor) {
  auto result = std::make_shared<PyExecuTorchResult>(tensor);
  py::object python_result = py::cast(result);
  if (portable_tensor_output() == PortableTensorOutput::ExecuTorch) {
    return python_result;
  }

  const auto modules = py::module_::import("sys").attr("modules");
  if (!modules.contains("torch")) {
    throw std::runtime_error(
        "PyTorch tensor output was requested, but torch is not imported. "
        "Import torch before executing the portable bindings or set "
        "EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT=executorch before importing "
        "them to receive ExecuTorchResult outputs.");
  }

  const auto torch_module = modules["torch"];
  const auto dtype = torch_module.attr(result->torch_dtype_name());
  if (result->nbytes() == 0) {
    return torch_module.attr("empty")(
        result->shape(), py::arg("dtype") = dtype);
  }
  return torch_module
      .attr("frombuffer")(python_result, py::arg("dtype") = dtype)
      .attr("as_strided")(result->shape(), result->element_strides());
}
#endif

void write_data_to_file(const std::string& path, void* buf, size_t size) {
  FILE* f = fopen(path.c_str(), "w+");
  if (!f) {
    throw std::runtime_error(
        "Failed to open file " + path + ": " + strerror(errno));
  }
  size_t num_written = fwrite(buf, 1, size, f);
  if (num_written != size) {
    fclose(f);
    throw std::runtime_error("Failed to write etdump to file " + path);
  }
  int err = fclose(f);
  if (err) {
    throw std::runtime_error(
        "Failed to close etdump file " + path + ": " + strerror(err));
  }
}

void setup_output_storage(
    Method& method,
    const std::vector<Span<uint8_t>>& output_storages) {
  if (output_storages.size() != method.outputs_size()) {
    THROW_IF_ERROR(
        Error::InvalidArgument,
        "number of output storages %zu does not match number of outputs %zu",
        output_storages.size(),
        method.outputs_size());
  }
  for (size_t i = 0; i < output_storages.size(); ++i) {
    if (output_storages[i].size() == 0) {
      // Skip empty output storages, this would happen for non-tensor outputs
      // and memory planned outputs.
      continue;
    }
    Error output_status = method.set_output_data_ptr(
        output_storages[i].data(), output_storages[i].size(), i);
    // We already should be skipping non-tensor outputs, and memory planned
    // outputs so any error is real.
    THROW_IF_ERROR(
        output_status,
        "set_output_data_ptr failed for output %zu with error 0x%" PRIx32,
        i,
        static_cast<uint32_t>(output_status));
  }
}

inline std::unique_ptr<DataLoader> loader_from_buffer(
    const void* ptr,
    size_t ptr_len) {
  return std::make_unique<BufferDataLoader>(ptr, ptr_len);
}

inline std::unique_ptr<DataLoader> loader_from_file(const std::string& path) {
  Result<MmapDataLoader> res = MmapDataLoader::from(
      path.c_str(), MmapDataLoader::MlockConfig::UseMlockIgnoreErrors);
  THROW_IF_ERROR(
      res.error(),
      "Failed to create MmapDataLoader from file %s, error: 0x:%" PRIx32,
      path.c_str(),
      static_cast<uint32_t>(res.error()));

  return std::make_unique<MmapDataLoader>(std::move(res.get()));
}

inline std::unique_ptr<Module> load_module_from_buffer(
    const void* ptr,
    size_t ptr_len,
    std::optional<const void*> data_map_ptr,
    std::optional<size_t> data_map_len,
    std::unique_ptr<runtime::EventTracer> event_tracer,
    Program::Verification program_verification) {
  EXECUTORCH_SCOPE_PROF("load_module_from_buffer");
  auto loader = loader_from_buffer(ptr, ptr_len);

  if (data_map_ptr.has_value() && data_map_len.has_value()) {
    auto data_map_loader =
        loader_from_buffer(data_map_ptr.value(), data_map_len.value());
    return std::make_unique<Module>(
        std::move(loader),
        nullptr, // memory_allocator
        nullptr, // temp_allocator
        std::move(event_tracer), // event_tracer
        std::move(data_map_loader)); // data_map_loader
  }

  return std::make_unique<Module>(
      std::move(loader),
      nullptr, // memory_allocator
      nullptr, // temp_allocator
      std::move(event_tracer), // event_tracer
      nullptr); // data_map_loader
}

inline std::unique_ptr<Module> load_module_from_file(
    const std::string& program_path,
    std::optional<const std::string>& data_map_path,
    std::unique_ptr<runtime::EventTracer> event_tracer,
    Program::Verification program_verification) {
  EXECUTORCH_SCOPE_PROF("load_module_from_file");

  auto program_loader = loader_from_file(program_path);
  if (data_map_path.has_value()) {
    auto data_map_loader = loader_from_file(data_map_path.value());
    return std::make_unique<Module>(
        std::move(program_loader),
        nullptr, // memory_allocator
        nullptr, // temp_allocator
        std::move(event_tracer), // event_tracer
        std::move(data_map_loader)); // data_map_loader
  }
  return std::make_unique<Module>(
      std::move(program_loader),
      nullptr, // memory_allocator
      nullptr, // temp_allocator
      std::move(event_tracer), // event_tracer
      nullptr); // data_map_loader
}

inline std::unique_ptr<Module> load_module_from_buffer_with_data_file(
    const void* ptr,
    size_t ptr_len,
    const std::string& data_map_path,
    std::unique_ptr<runtime::EventTracer> event_tracer,
    Program::Verification program_verification) {
  auto program_loader = loader_from_buffer(ptr, ptr_len);
  auto data_loader = loader_from_file(data_map_path);
  return std::make_unique<Module>(
      std::move(program_loader),
      nullptr, // memory_allocator
      nullptr, // temp_allocator
      std::move(event_tracer), // event_tracer
      std::move(data_loader));
}

inline std::unique_ptr<Module> load_module_from_data_loader(
    std::shared_ptr<PyDataLoader> loader,
    std::optional<const std::string> data_map_path,
    std::unique_ptr<runtime::EventTracer> event_tracer) {
  EXECUTORCH_SCOPE_PROF("load_module_from_data_loader");

  if (data_map_path.has_value()) {
    auto data_map_loader = loader_from_file(data_map_path.value());
    return std::make_unique<Module>(
        loader->make_delegating_loader(),
        nullptr, // memory_allocator
        nullptr, // temp_allocator
        std::move(event_tracer), // event_tracer
        std::move(data_map_loader)); // data_map_loader
  }
  return std::make_unique<Module>(
      loader->make_delegating_loader(),
      nullptr, // memory_allocator
      nullptr, // temp_allocator
      std::move(event_tracer), // event_tracer
      nullptr); // data_map_loader
}

inline py::list get_outputs_as_py_list(
    const std::vector<EValue>& outputs,
    bool clone_outputs = true) {
  const auto outputs_size = outputs.size();
  py::list list(outputs_size);
  for (size_t i = 0; i < outputs_size; ++i) {
    auto& v = outputs[i];
    if (Tag::None == v.tag) {
      list[i] = py::none();
    } else if (Tag::Int == v.tag) {
      list[i] = py::cast(v.toInt());
    } else if (Tag::Double == v.tag) {
      list[i] = py::cast(v.toDouble());
    } else if (Tag::Bool == v.tag) {
      list[i] = py::cast(v.toBool());
    } else if (Tag::String == v.tag) {
      list[i] = py::cast(std::string(v.toString().data()));
    } else if (Tag::Tensor == v.tag) {
#ifdef USE_ATEN_LIB
      // Clone so the outputs in python do not share a lifetime with the
      // module object
      if (clone_outputs) {
        list[i] = py::cast(v.toTensor().clone());
      } else {
        list[i] = py::cast(v.toTensor());
      }
#else
      (void)clone_outputs;
      list[i] = portable_tensor_result(v.toTensor());
#endif
    } else {
      ET_ASSERT_UNREACHABLE_MSG("Invalid model output type");
    }
  }
  return list;
}

static constexpr size_t kDEFAULT_BUNDLED_INPUT_POOL_SIZE = 16 * 1024U;

struct PyBundledModule : public BundledModule {
  explicit PyBundledModule(
      const py::bytes& buffer,
      uint32_t bundled_input_pool_size)
      : BundledModule(buffer.cast<std::string_view>().data()),
        bundled_program_ptr_(buffer),
        program_ptr_(static_cast<const void*>(
            bundled_program_flatbuffer::GetBundledProgram(
                get_bundled_program_ptr())
                ->program()
                ->data())),
        program_len_(bundled_program_flatbuffer::GetBundledProgram(
                         get_bundled_program_ptr())
                         ->program()
                         ->size()) {}

  static std::unique_ptr<PyBundledModule> load_from_buffer(
      const py::bytes& buffer,
      uint32_t bundled_input_pool_size) {
    return std::make_unique<PyBundledModule>(buffer, bundled_input_pool_size);
  }

  const void* get_bundled_program_ptr() {
    return bundled_program_ptr_.cast<std::string_view>().data();
  }

  const void* get_program_ptr() {
    return program_ptr_;
  }

  size_t get_program_len() {
    return program_len_;
  }

  py::list verify_result_with_bundled_expected_output(
      const std::string& method_name,
      size_t testset_idx,
      double rtol = 1e-5,
      double atol = 1e-8) {
    // Execute the method
    auto result = BundledModule::execute(method_name, testset_idx);
    if (!result.ok()) {
      THROW_IF_ERROR(
          result.error(),
          "Method execution failed with status 0x%" PRIx32,
          static_cast<uint32_t>(result.error()));
    }

    // Convert outputs to py::list
    const auto& outputs = result.get();
    py::list py_outputs = get_outputs_as_py_list(outputs);

    Error status = BundledModule::verify_method_outputs(
        method_name, testset_idx, rtol, atol);
    THROW_IF_ERROR(
        status,
        "Result verification failed with status %" PRIu32,
        static_cast<uint32_t>(status));
    return py_outputs;
  }

 private:
  // Store the bytes object instead of a raw pointer so that this module will
  // keep the bytes alive.
  const py::bytes bundled_program_ptr_;
  const void* program_ptr_;
  size_t program_len_;
};

// Program points to DataLoader so bundle them up into a struct to ensure that
// it stays alive.
struct ProgramState final {
  std::unique_ptr<DataLoader> loader_;
  std::unique_ptr<Program> program_;
  // Owned here rather than by PyProgram, beside the loader it reads
  // through, because a Method borrows the map and outlives the call that
  // loaded it. Both stay alive as long as any method does.
  std::unique_ptr<DataLoader> data_map_loader_;
  std::unique_ptr<FlatTensorDataMap> data_map_;

  explicit ProgramState(
      std::unique_ptr<DataLoader> loader,
      std::unique_ptr<Program> program,
      std::unique_ptr<DataLoader> data_map_loader = nullptr,
      std::unique_ptr<FlatTensorDataMap> data_map = nullptr)
      : loader_(std::move(loader)),
        program_(std::move(program)),
        data_map_loader_(std::move(data_map_loader)),
        data_map_(std::move(data_map)) {}
  ProgramState(const ProgramState&) = delete;
  ProgramState& operator=(const ProgramState&) = delete;
  ProgramState(ProgramState&&) = default;
  ProgramState& operator=(ProgramState&&) = default;
};

/// Expose a subset of TensorInfo information to python.
struct PyTensorInfo final {
  explicit PyTensorInfo(
      std::shared_ptr<Module> module,
      torch::executor::TensorInfo info)
      : module_(std::move(module)), state_(nullptr), info_(info) {}

  explicit PyTensorInfo(
      std::shared_ptr<ProgramState> state,
      torch::executor::TensorInfo info)
      : module_(nullptr), state_(std::move(state)), info_(info) {}

  py::tuple sizes() const {
    const auto shape = info_.sizes();
    py::tuple tup(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      tup[i] = py::cast(shape[i]);
    }
    return tup;
  }

  int8_t dtype() const {
    return static_cast<
        std::underlying_type<executorch::aten::ScalarType>::type>(
        info_.scalar_type());
  }

  bool is_memory_planned() const {
    return info_.is_memory_planned();
  }

  size_t nbytes() const {
    return info_.nbytes();
  }

  std::string repr() const {
    std::string size_str = "[";
    for (const auto& d : info_.sizes()) {
      size_str.append(std::to_string(d));
      size_str.append(", ");
    }
    if (size_str.length() >= 2) {
      // Pop the last two characters (command and space) and add close bracket.
      size_str.pop_back();
      size_str.pop_back();
    }
    size_str.append("]");
    return "TensorInfo(sizes=" + size_str + ", dtype=" +
        std::string(executorch::runtime::toString(info_.scalar_type())) +
        ", is_memory_planned=" +
        (info_.is_memory_planned() ? "True" : "False") +
        ", nbytes=" + std::to_string(info_.nbytes()) + ")";
  }

 private:
  // TensorInfo relies on either a module or program to be alive.
  std::shared_ptr<Module> module_;
  std::shared_ptr<ProgramState> state_;
  torch::executor::TensorInfo info_;
};

/// Expose a subset of MethodMeta information to python.
struct PyMethodMeta final {
  explicit PyMethodMeta(
      std::shared_ptr<Module> module,
      torch::executor::MethodMeta meta)
      : module_(std::move(module)), state_(nullptr), meta_(meta) {}

  explicit PyMethodMeta(
      std::shared_ptr<ProgramState> state,
      torch::executor::MethodMeta meta)
      : module_(nullptr), state_(std::move(state)), meta_(meta) {}

  const char* name() const {
    return meta_.name();
  }

  size_t num_inputs() const {
    return meta_.num_inputs();
  }

  std::unique_ptr<PyTensorInfo> input_tensor_meta(size_t index) const {
    const auto result = meta_.input_tensor_meta(index);
    THROW_INDEX_IF_ERROR(
        result.error(), "Cannot get input tensor meta at %zu", index);
    if (module_) {
      return std::make_unique<PyTensorInfo>(module_, result.get());
    } else {
      return std::make_unique<PyTensorInfo>(state_, result.get());
    }
  }

  size_t num_outputs() const {
    return meta_.num_outputs();
  }

  std::unique_ptr<PyTensorInfo> output_tensor_meta(size_t index) const {
    const auto result = meta_.output_tensor_meta(index);
    THROW_INDEX_IF_ERROR(
        result.error(), "Cannot get output tensor meta at %zu", index);
    if (module_) {
      return std::make_unique<PyTensorInfo>(module_, result.get());
    } else {
      return std::make_unique<PyTensorInfo>(state_, result.get());
    }
  }

  size_t num_attributes() const {
    return meta_.num_attributes();
  }

  std::unique_ptr<PyTensorInfo> attribute_tensor_meta(size_t index) const {
    const auto result = meta_.attribute_tensor_meta(index);
    THROW_INDEX_IF_ERROR(
        result.error(), "Cannot get attribute tensor meta at %zu", index);
    if (module_) {
      return std::make_unique<PyTensorInfo>(module_, result.get());
    } else {
      return std::make_unique<PyTensorInfo>(state_, result.get());
    }
  }

  py::str repr() const {
    py::list input_meta_strs;
    for (size_t i = 0; i < meta_.num_inputs(); ++i) {
      input_meta_strs.append(py::str(input_tensor_meta(i)->repr()));
    }
    py::list output_meta_strs;
    for (size_t i = 0; i < meta_.num_outputs(); ++i) {
      auto output_tag_res = meta_.output_tag(i);
      THROW_INDEX_IF_ERROR(
          output_tag_res.error(), "Cannot get Tag for output at %zu", i);
      if (output_tag_res.get() == Tag::Tensor) {
        output_meta_strs.append(py::str(output_tensor_meta(i)->repr()));
      } else {
        output_meta_strs.append(
            py::str(runtime::tag_to_string(output_tag_res.get())));
      }
    }
    // Add quotes to be more similar to Python's repr for strings.
    py::str format =
        "MethodMeta(name='{}', num_inputs={}, input_tensor_meta={}, num_outputs={}, output_tensor_meta={})";
    return format.format(
        std::string(meta_.name()),
        std::to_string(meta_.num_inputs()),
        input_meta_strs,
        std::to_string(meta_.num_outputs()),
        output_meta_strs);
  }

 private:
  // Must keep the either the Module or Program object alive or else the meta
  // object is invalidated.
  std::shared_ptr<Module> module_;
  std::shared_ptr<ProgramState> state_;
  torch::executor::MethodMeta meta_;
};

struct PyModule final {
  explicit PyModule(
      const py::bytes& buffer,
      std::optional<const py::bytes> data_map_buffer,
      bool enable_etdump,
      size_t debug_buffer_size = 0,
      Program::Verification program_verification =
          Program::Verification::InternalConsistency)
      : debug_buffer_size_(debug_buffer_size),
        module_(load_module_from_buffer(
            buffer.cast<std::string_view>().data(),
            py::len(buffer),
            data_map_buffer.has_value()
                ? std::optional<const void*>(
                      data_map_buffer.value().cast<std::string_view>().data())
                : std::nullopt,
            data_map_buffer.has_value()
                ? std::optional<size_t>(py::len(data_map_buffer.value()))
                : std::nullopt,
            setup_event_tracer(enable_etdump, debug_buffer_size),
            program_verification)) {}

  explicit PyModule(
      const void* ptr,
      size_t ptr_len,
      std::optional<const void*> data_map_ptr,
      std::optional<size_t> data_map_ptr_len,
      bool enable_etdump,
      size_t debug_buffer_size = 0,
      Program::Verification program_verification =
          Program::Verification::InternalConsistency)
      : debug_buffer_size_(debug_buffer_size),
        module_(load_module_from_buffer(
            ptr,
            ptr_len,
            data_map_ptr,
            data_map_ptr_len,
            setup_event_tracer(enable_etdump, debug_buffer_size),
            program_verification)) {}

  explicit PyModule(
      const void* ptr,
      size_t ptr_len,
      const std::string& data_path,
      bool enable_etdump,
      size_t debug_buffer_size = 0,
      Program::Verification program_verification =
          Program::Verification::InternalConsistency)
      : debug_buffer_size_(debug_buffer_size),
        module_(load_module_from_buffer_with_data_file(
            ptr,
            ptr_len,
            data_path,
            setup_event_tracer(enable_etdump, debug_buffer_size),
            program_verification)) {}

  explicit PyModule(
      const std::string& program_path,
      std::optional<const std::string>& data_path,
      bool enable_etdump,
      size_t debug_buffer_size = 0,
      Program::Verification program_verification =
          Program::Verification::InternalConsistency)
      : debug_buffer_size_(debug_buffer_size),
        module_(load_module_from_file(
            program_path,
            data_path,
            setup_event_tracer(enable_etdump, debug_buffer_size),
            program_verification)) {}

  explicit PyModule(
      std::shared_ptr<PyDataLoader> loader,
      std::optional<const std::string> data_path,
      bool enable_etdump,
      size_t debug_buffer_size = 0)
      : debug_buffer_size_(debug_buffer_size),
        module_(load_module_from_data_loader(
            std::move(loader),
            data_path,
            setup_event_tracer(enable_etdump, debug_buffer_size))) {}

  PyModule(const PyModule&) = delete;
  PyModule& operator=(const PyModule&) = delete;
  PyModule(PyModule&&) = default;
  PyModule& operator=(PyModule&&) = default;

  // Module is only valid as long as the python buffer is alive.
  static std::unique_ptr<PyModule> load_from_buffer(
      const py::bytes& buffer,
      std::optional<const py::bytes> data_map_buffer,
      bool enable_etdump,
      size_t debug_buffer_size = 0,
      Program::Verification program_verification =
          Program::Verification::InternalConsistency) {
    return std::make_unique<PyModule>(
        buffer,
        data_map_buffer,
        enable_etdump,
        debug_buffer_size,
        program_verification);
  }

  static std::unique_ptr<PyModule> load_from_file(
      const std::string& program_path,
      std::optional<const std::string>& data_path,
      bool enable_etdump,
      size_t debug_buffer_size = 0,
      Program::Verification program_verification =
          Program::Verification::InternalConsistency) {
    return std::make_unique<PyModule>(
        program_path,
        data_path,
        enable_etdump,
        debug_buffer_size,
        program_verification);
  }

  // Load with data as a buffer.
  static std::unique_ptr<PyModule> load_from_bundled_program(
      PyBundledModule& m,
      std::optional<const py::bytes> data_map_buffer,
      bool enable_etdump,
      size_t debug_buffer_size = 0) {
    std::optional<const void*> data_map_ptr = std::nullopt;
    std::optional<size_t> data_map_len = std::nullopt;

    if (data_map_buffer.has_value()) {
      data_map_ptr = data_map_buffer.value().cast<std::string_view>().data();
      data_map_len = py::len(data_map_buffer.value());
    }

    return std::make_unique<PyModule>(
        m.get_program_ptr(),
        m.get_program_len(),
        data_map_ptr,
        data_map_len,
        enable_etdump,
        debug_buffer_size,
        Program::Verification::InternalConsistency);
  }

  // Load with data as a file.
  static std::unique_ptr<PyModule> load_from_bundled_program(
      PyBundledModule& m,
      const std::string& data_path,
      bool enable_etdump,
      size_t debug_buffer_size = 0) {
    return std::make_unique<PyModule>(
        m.get_program_ptr(),
        m.get_program_len(),
        data_path,
        enable_etdump,
        debug_buffer_size,
        Program::Verification::InternalConsistency);
  }

  // Load from an external data loader.
  // This allows external libraries (like PTEZ) to provide custom data loaders.
  static std::unique_ptr<PyModule> load_from_data_loader(
      std::shared_ptr<PyDataLoader> loader,
      std::optional<const std::string> data_path,
      bool enable_etdump,
      size_t debug_buffer_size = 0) {
    return std::make_unique<PyModule>(
        std::move(loader), data_path, enable_etdump, debug_buffer_size);
  }

  py::list run_method(
      const std::string& method_name,
      const py::object& python_inputs,
      bool clone_outputs = true) {
    const auto inputs = normalize_inputs(python_inputs);
    const auto inputs_size = py::len(inputs);
    const auto method_meta_result = module_->method_meta(method_name);
    THROW_IF_ERROR(
        method_meta_result.error(),
        "Failed to get metadata for method %s",
        method_name.c_str());
    const auto method_meta = method_meta_result.get();
    std::vector<EValue> cpp_inputs;
    cpp_inputs.reserve(inputs_size);
    std::vector<std::shared_ptr<BufferTensor>> buffer_inputs;
    buffer_inputs.reserve(inputs_size);
#ifndef USE_ATEN_LIB
    std::vector<std::shared_ptr<TorchTensorView>> torch_inputs;
    torch_inputs.reserve(inputs_size);
#endif
    bool saw_buffer = false;
    bool saw_torch = false;

#ifndef USE_ATEN_LIB // Portable mode
    // So the ETensors and their metadata stay in scope for
    // Module->run_method.
    std::vector<torch::executor::TensorImpl> input_tensors;
    std::vector<std::vector<torch::executor::Tensor::SizesType>> input_sizes;
    std::vector<std::vector<torch::executor::Tensor::StridesType>>
        input_strides;
    std::vector<std::vector<torch::executor::Tensor::DimOrderType>>
        input_dim_order;
    // We store pointers to these vector elements so important to reserve so
    // that we don't lose those on a vector resize. Don't need to do this for
    // the others since they are vectors of vectors, and we don't store a
    // pointer to the root level vector data.
    input_tensors.reserve(inputs_size);
#endif

    // Convert python objects into EValues.
    for (size_t i = 0; i < inputs_size; ++i) {
      auto python_input = inputs[i];
      const std::string& type_str = py::str(python_input.get_type());
      if (is_torch_tensor(python_input)) {
        if (saw_buffer) {
          throw py::type_error(
              "A call cannot mix buffer and torch tensor inputs");
        }
        saw_torch = true;
#ifdef USE_ATEN_LIB
        auto at_tensor = python_input.cast<at::Tensor>();
        std::vector<int> tensor_sizes(
            at_tensor.sizes().begin(), at_tensor.sizes().end());
        std::vector<int> tensor_strides(
            at_tensor.strides().begin(), at_tensor.strides().end());
        validate_tensor_input(
            method_meta,
            i,
            runtime_scalar_type(at_tensor),
            tensor_sizes,
            tensor_strides);
        (void)mutable_tensor_data_ptr_no_cow(at_tensor);
        cpp_inputs.emplace_back(at_tensor);
#else
        torch_inputs.push_back(std::make_shared<TorchTensorView>(python_input));
        const auto& tensor = torch_inputs.back();
        validate_tensor_input(
            method_meta,
            i,
            tensor->scalar_type(),
            tensor->sizes(),
            tensor->strides());
        input_sizes.emplace_back(
            tensor->sizes().begin(), tensor->sizes().end());
        input_strides.emplace_back(
            tensor->strides().begin(), tensor->strides().end());
        input_dim_order.emplace_back(
            tensor->dim_order().begin(), tensor->dim_order().end());
        input_tensors.emplace_back(
            tensor->scalar_type(),
            input_sizes.back().size(),
            input_sizes.back().data(),
            tensor->data(),
            input_dim_order.back().data(),
            input_strides.back().data(),
            torch::executor::TensorShapeDynamism::STATIC);
        cpp_inputs.emplace_back(torch::executor::Tensor(&input_tensors.back()));
#endif
      } else if (py::isinstance<py::none>(python_input)) {
        cpp_inputs.push_back(EValue());
      } else if (py::isinstance<py::bool_>(python_input)) {
        cpp_inputs.push_back(EValue(py::cast<bool>(python_input)));
      } else if (py::isinstance<py::int_>(python_input)) {
        cpp_inputs.push_back(EValue(py::cast<int64_t>(python_input)));
      } else if (py::isinstance<py::float_>(python_input)) {
        cpp_inputs.push_back(EValue(py::cast<double>(python_input)));
      } else if (PyObject_CheckBuffer(python_input.ptr())) {
        if (saw_torch) {
          throw py::type_error(
              "A call cannot mix buffer and torch tensor inputs");
        }
        saw_buffer = true;
        buffer_inputs.push_back(std::make_shared<BufferTensor>(python_input));
        const auto& buffer = buffer_inputs.back();
        validate_tensor_input(
            method_meta,
            i,
            buffer->scalar_type(),
            buffer->sizes(),
            buffer->strides());
#ifdef USE_ATEN_LIB
        auto at_tensor = at::from_blob(
            buffer->data(),
            std::vector<int64_t>(
                buffer->sizes().begin(), buffer->sizes().end()),
            std::vector<int64_t>(
                buffer->strides().begin(), buffer->strides().end()),
            at::TensorOptions().dtype(buffer->scalar_type()));
        cpp_inputs.emplace_back(at_tensor);
#else
        input_sizes.emplace_back(
            buffer->sizes().begin(), buffer->sizes().end());
        input_strides.emplace_back(
            buffer->strides().begin(), buffer->strides().end());
        input_dim_order.emplace_back(
            buffer->dim_order().begin(), buffer->dim_order().end());
        input_tensors.emplace_back(
            buffer->scalar_type(),
            input_sizes.back().size(),
            input_sizes.back().data(),
            buffer->data(),
            input_dim_order.back().data(),
            input_strides.back().data(),
            torch::executor::TensorShapeDynamism::STATIC);
        cpp_inputs.emplace_back(torch::executor::Tensor(&input_tensors.back()));
#endif
      } else {
        throw std::runtime_error(
            "Unsupported python type " + type_str +
            ". Ensure that inputs are passed as a flat list of tensors.");
      }
    }

    // Set up output storage before execution.
    allocate_output_tensors(method_name);
    auto outputs = module_->execute(method_name, cpp_inputs);
    THROW_IF_ERROR(
        outputs.error(),
        "Failed to execute method %s, error: 0x%" PRIx32,
        method_name.c_str(),
        static_cast<uint32_t>(outputs.error()));

    // Retrieve outputs
    return get_outputs_as_py_list(outputs.get(), clone_outputs);
  }

  py::list forward(const py::object& inputs, bool clone_outputs = true) {
    return run_method("forward", inputs, clone_outputs);
  }

  bool has_etdump() {
    ETDumpGen* etdump = dynamic_cast<ETDumpGen*>(module_->event_tracer());
    return etdump != nullptr;
  }

  void write_etdump_result_to_file(
      const std::string& path,
      const py::object& debug_buffer_path) {
    if (!has_etdump()) {
      throw std::runtime_error("No etdump found");
    }
    ETDumpGen* etdump = dynamic_cast<ETDumpGen*>(module_->event_tracer());
    etdump_result result = etdump->get_etdump_data();
    if (result.buf != nullptr && result.size > 0) {
      write_data_to_file(path, result.buf, result.size);
      free(result.buf);
      if (py::isinstance<py::str>(debug_buffer_path)) {
        // Also write out the debug buffer to a separate file if requested.
        std::string debug_buffer_path_str =
            py::cast<std::string>(debug_buffer_path);
        if (debug_buffer_ && debug_buffer_size_ > 0) {
          write_data_to_file(
              debug_buffer_path_str, debug_buffer_.get(), debug_buffer_size_);
        }
      }
    } else {
      ET_LOG(
          Info,
          "No etdump data found, try rebuilding with "
          "the CMake option EXECUTORCH_ENABLE_EVENT_TRACER or with "
          "buck run --config executorch.event_tracer_enabled=true");
    }
  }

  py::list plan_execute(
      const std::string method_name,
      bool clone_outputs = true) {
    auto status = module_->load_method(method_name);

    THROW_IF_ERROR(
        status,
        "executing execution plan for method 'load' failed with error: 0x%" PRIx32,
        static_cast<uint32_t>(status));
    auto output = module_->execute(method_name.c_str());
    THROW_IF_ERROR(
        output.error(),
        "executing execution plan for method 'forward' failed with error: 0x%" PRIx32,
        static_cast<uint32_t>(output.error()));
    return get_outputs_as_py_list(output.get(), clone_outputs);
  }

  std::unique_ptr<PyMethodMeta> method_meta(const std::string method_name) {
    auto method_data = module_->method_meta(method_name);
    THROW_IF_ERROR(
        method_data.error(),
        "failed to retrieve method_meta for method %s, error 0x%" PRIx32,
        method_name.c_str(),
        static_cast<uint32_t>(method_data.error()));
    return std::make_unique<PyMethodMeta>(module_, method_data.get());
  }

  std::vector<std::string> method_names() {
    auto result = module_->method_names();
    THROW_IF_ERROR(
        result.error(),
        "Failed to get method names, error: 0x%" PRIx32,
        static_cast<uint32_t>(result.error()));
    const auto& method_set = result.get();
    return std::vector<std::string>(method_set.begin(), method_set.end());
  }

 private:
  // Hold onto the debug_buffer_ for the event_tracer.
  std::unique_ptr<uint8_t[]> debug_buffer_;
  size_t debug_buffer_size_;

  std::shared_ptr<Module> module_;
  // Need to keep-alive output tensors until they can be compared in case of
  // bundled programs.
  std::vector<std::optional<TensorPtr>> output_tensors_;

  // Set debug buffer for potential event tracer.
  std::unique_ptr<torch::executor::ETDumpGen> setup_event_tracer(
      bool enable_etdump,
      size_t debug_buffer_size) {
    std::unique_ptr<torch::executor::ETDumpGen> event_tracer = enable_etdump
        ? std::make_unique<torch::executor::ETDumpGen>()
        : nullptr;
    if (enable_etdump && debug_buffer_size > 0) {
      debug_buffer_ = std::make_unique<uint8_t[]>(debug_buffer_size);
      debug_buffer_size_ = debug_buffer_size;
      event_tracer->set_debug_buffer(
          Span<uint8_t>(debug_buffer_.get(), debug_buffer_size));
      event_tracer->set_event_tracer_debug_level(
          EventTracerDebugLogLevel::kIntermediateOutputs);
    }
    return event_tracer;
  }

  // Allocate output tensors when they are not memory planned.
  void allocate_output_tensors(const std::string& method_name) {
    auto method_meta_result = module_->method_meta(method_name);
    THROW_IF_ERROR(
        method_meta_result.error(),
        "Failed to get method_meta for %s, error: 0x%" PRIx32,
        method_name.c_str(),
        static_cast<uint32_t>(method_meta_result.error()));

    auto method_meta = method_meta_result.get();
    const auto num_outputs = method_meta.num_outputs();

    // Create a buffer for each output tensor. Memory planned outputs and non
    // tensor outputs get an empty buffer in this list which is ignored later.
    output_tensors_.clear();
    output_tensors_.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
      auto output_type = method_meta.output_tag(i);
      THROW_IF_ERROR(
          output_type.error(), "Failed to get output type for output %zu", i);
      if (output_type.get() != Tag::Tensor) {
        // Skip allocating storage for non-tensor outputs.
        output_tensors_.emplace_back(std::nullopt);
        continue;
      }
      const auto& output_tensor_meta = method_meta.output_tensor_meta(i);
      THROW_IF_ERROR(
          output_tensor_meta.error(),
          "Failed to get output tensor meta for output %zu",
          i);
      if (output_tensor_meta.get().is_memory_planned()) {
        // Skip allocating storage for planned memory outputs.
        output_tensors_.emplace_back(std::nullopt);
        continue;
      }
      TensorPtr tensor_ptr = make_tensor_ptr(
          std::vector<executorch::aten::SizesType>(
              output_tensor_meta->sizes().begin(),
              output_tensor_meta->sizes().end()),
          std::vector<uint8_t>(output_tensor_meta->nbytes()),
          output_tensor_meta->scalar_type());
      output_tensors_.emplace_back(std::move(tensor_ptr));
    }

    for (size_t i = 0; i < output_tensors_.size(); ++i) {
      if (output_tensors_[i].has_value()) {
        // Set output tensors on module.
        auto status = module_->set_output(method_name, output_tensors_[i], i);
        THROW_IF_ERROR(
            status,
            "Failed to set output for method %s, error: 0x%" PRIx32,
            method_name.c_str(),
            static_cast<uint32_t>(status));
      }
    }
  }
};

inline std::shared_ptr<ProgramState> load_program(
    std::unique_ptr<DataLoader> loader,
    Program::Verification program_verification,
    std::optional<const std::string> data_path = std::nullopt) {
  Result<Program> res = Program::load(loader.get(), program_verification);
  THROW_IF_ERROR(
      res.error(),
      "Failed to load program, error: 0x:%" PRIx32,
      static_cast<uint32_t>(res.error()));
  // A program whose weights live outside the pte names them through a data
  // map. The CUDA delegate emits exactly that: a pte holding the compiled
  // kernels and a separate file holding the weights, so loading the pte alone
  // produces a program that fails when a method runs rather than when it
  // loads.
  std::unique_ptr<DataLoader> data_map_loader;
  std::unique_ptr<FlatTensorDataMap> data_map;
  if (data_path.has_value()) {
    data_map_loader = loader_from_file(data_path.value());
    Result<FlatTensorDataMap> map_res =
        FlatTensorDataMap::load(data_map_loader.get());
    THROW_IF_ERROR(
        map_res.error(),
        "Failed to load data map from %s, error: 0x:%" PRIx32,
        data_path.value().c_str(),
        static_cast<uint32_t>(map_res.error()));
    data_map = std::make_unique<FlatTensorDataMap>(std::move(map_res.get()));
  }
  return std::make_shared<ProgramState>(
      std::move(loader),
      std::make_unique<Program>(std::move(res.get())),
      std::move(data_map_loader),
      std::move(data_map));
}

/// A wrapper/util class for executorch memory allocations/manager.
class ProgramMemory {
 public:
  /// `devices` is empty when every buffer is on the host, which keeps
  /// `MemoryManager::has_device_memory()` false for CPU-only programs.
  /// Otherwise it holds one entry per buffer, indexed like `sizes`.
  ///
  /// Members initialize in declaration order and each one reads the members
  /// declared before it, so that order is load-bearing. Device buffers come
  /// first so that a device that is missing or out of memory throws before the
  /// host arenas are allocated and zero-filled, rather than after.
  ProgramMemory(
      std::vector<int64_t>&& sizes,
      std::vector<runtime::etensor::Device>&& devices)
      : runtime_allocator_(),
        planned_sizes_(std::move(sizes)),
        planned_devices_(std::move(devices)),
        device_buffers_(allocate_device_buffers()),
        non_const_buffers_(allocate_host_buffers()),
        non_const_spans_(create_non_const_spans()),
        non_const_allocator_(create_non_const_allocator()),
        mem_manager_(
            &const_allocator_,
            &non_const_allocator_,
            &runtime_allocator_,
            &temp_allocator_) {}

  explicit ProgramMemory(std::vector<int64_t>&& sizes)
      : ProgramMemory(std::move(sizes), {}) {}

  /// Returns a pointer to the internal memory manager, the Memory instance
  /// must outlive this pointer.
  MemoryManager* mem_manager() {
    return &mem_manager_;
  }

  ProgramMemory(const ProgramMemory&) = delete;
  ProgramMemory& operator=(const ProgramMemory&) = delete;

 private:
  MemoryAllocator const_allocator_{MemoryAllocator(0, nullptr)};

  MallocMemoryAllocator runtime_allocator_;

  MallocMemoryAllocator temp_allocator_{};

  std::vector<int64_t> planned_sizes_;

  std::vector<runtime::etensor::Device> planned_devices_;

  // Backs device-tagged buffers; the entry is empty for a CPU-tagged buffer.
  // Parallel to non_const_buffers_ so both index by planned buffer id. Empty
  // for an all-host program.
  std::vector<DeviceMemoryBuffer> device_buffers_;

  // Backs CPU-tagged buffers; the entry is empty for a device-tagged buffer.
  std::vector<std::vector<uint8_t>> non_const_buffers_;

  std::vector<Span<uint8_t>> non_const_spans_;

  HierarchicalAllocator non_const_allocator_;

  MemoryManager mem_manager_;

  bool is_device_buffer(size_t index) const {
    return index < planned_devices_.size() && !planned_devices_[index].is_cpu();
  }

  std::vector<std::vector<uint8_t>> allocate_host_buffers() {
    std::vector<std::vector<uint8_t>> result;
    result.reserve(planned_sizes_.size());
    for (size_t i = 0; i < planned_sizes_.size(); ++i) {
      if (is_device_buffer(i)) {
        result.emplace_back();
      } else {
        result.emplace_back(planned_sizes_[i]);
      }
    }
    return result;
  }

  std::vector<DeviceMemoryBuffer> allocate_device_buffers() {
    std::vector<DeviceMemoryBuffer> result;
    if (planned_devices_.empty()) {
      return result;
    }
    // Both vectors are filled in lockstep today, so this only fires if a
    // future caller breaks that. HierarchicalAllocator aborts on a mismatch,
    // so check here instead, where a Python caller can catch it.
    THROW_IF_ERROR(
        planned_devices_.size() == planned_sizes_.size()
            ? Error::Ok
            : Error::InvalidArgument,
        "Have %zu planned buffer sizes but %zu device tags",
        planned_sizes_.size(),
        planned_devices_.size());
    result.reserve(planned_sizes_.size());
    for (size_t i = 0; i < planned_sizes_.size(); ++i) {
      if (!is_device_buffer(i)) {
        result.emplace_back();
        continue;
      }
      auto buffer = DeviceMemoryBuffer::create(
          planned_sizes_[i],
          planned_devices_[i].type(),
          planned_devices_[i].index());
      THROW_IF_ERROR(
          buffer.error(),
          "Failed to allocate %" PRId64 " bytes for buffer %zu on device %d:%d",
          planned_sizes_[i],
          i,
          static_cast<int>(planned_devices_[i].type()),
          static_cast<int>(planned_devices_[i].index()));
      result.emplace_back(std::move(buffer.get()));
    }
    return result;
  }

  std::vector<Span<uint8_t>> create_non_const_spans() {
    std::vector<Span<uint8_t>> result;
    result.reserve(planned_sizes_.size());
    for (size_t i = 0; i < planned_sizes_.size(); ++i) {
      if (is_device_buffer(i)) {
        result.push_back(device_buffers_[i].as_span());
      } else {
        result.push_back(
            {non_const_buffers_[i].data(), non_const_buffers_[i].size()});
      }
    }
    return result;
  }

  HierarchicalAllocator create_non_const_allocator() {
    Span<Span<uint8_t>> buffers(
        non_const_spans_.data(), non_const_spans_.size());
    return planned_devices_.empty()
        ? HierarchicalAllocator(buffers)
        : HierarchicalAllocator(
              buffers, {planned_devices_.data(), planned_devices_.size()});
  }
};

/// True if any of the method's memory-planned buffers must live off the host.
bool has_device_buffers(const MethodMeta& method_meta) {
  for (size_t i = 0; i < method_meta.num_memory_planned_buffers(); ++i) {
    auto device = method_meta.memory_planned_buffer_device(i);
    THROW_IF_ERROR(
        device.error(), "Failed to get device of planned buffer %zu", i);
    if (!device.get().is_cpu()) {
      return true;
    }
  }
  return false;
}

/// Arenas sized and placed for a single method, used when that method's
/// buffers cannot come from the program-wide host arenas. Returns nullptr when
/// every buffer is on the host, so one pass over the metadata answers both
/// whether the method needs its own arenas and how big they are.
std::shared_ptr<ProgramMemory> make_method_memory(
    const MethodMeta& method_meta) {
  const size_t num_buffers = method_meta.num_memory_planned_buffers();
  std::vector<int64_t> sizes;
  std::vector<runtime::etensor::Device> devices;
  sizes.reserve(num_buffers);
  devices.reserve(num_buffers);
  bool needs_device_memory = false;
  for (size_t i = 0; i < num_buffers; ++i) {
    auto size = method_meta.memory_planned_buffer_size(i);
    THROW_IF_ERROR(size.error(), "Failed to get size of planned buffer %zu", i);
    auto device = method_meta.memory_planned_buffer_device(i);
    THROW_IF_ERROR(
        device.error(), "Failed to get device of planned buffer %zu", i);
    needs_device_memory |= !device.get().is_cpu();
    sizes.push_back(size.get());
    devices.push_back(device.get());
  }
  if (!needs_device_memory) {
    return nullptr;
  }
  return std::make_shared<ProgramMemory>(std::move(sizes), std::move(devices));
}

struct PyMethod final {
  explicit PyMethod(
      std::shared_ptr<ProgramMemory> memory,
      std::shared_ptr<ProgramState> state,
      std::unique_ptr<Method> method)
      : memory_(std::move(memory)),
        state_(std::move(state)),
        method_(std::move(method)) {}

  void set_inputs(const py::object& python_inputs) {
    const auto inputs = normalize_inputs(python_inputs);
    const auto inputs_size = py::len(inputs);
    std::vector<EValue> cpp_inputs;
    cpp_inputs.reserve(inputs_size);
    std::vector<std::shared_ptr<BufferTensor>> buffer_inputs;
    buffer_inputs.reserve(inputs_size);
    std::vector<TensorPtr> buffer_tensor_ptrs;
    buffer_tensor_ptrs.reserve(inputs_size);
#ifndef USE_ATEN_LIB
    std::vector<std::shared_ptr<TorchTensorView>> torch_inputs;
    torch_inputs.reserve(inputs_size);
#endif
    bool saw_buffer = false;
    bool saw_torch = false;

#ifndef USE_ATEN_LIB // Portable mode
    // So the ETensors and their metadata stay in scope for
    // Module->set_inputs.
    std::vector<TensorPtr> input_tensors;
    // We store pointers to these vector elements so important to reserve so
    // that we don't lose those on a vector resize.
    input_tensors.reserve(inputs_size);
#endif

    // Convert python objects into EValues.
    for (size_t i = 0; i < inputs_size; ++i) {
      auto python_input = inputs[i];
      const std::string& type_str = py::str(python_input.get_type());
      if (is_torch_tensor(python_input)) {
        if (saw_buffer) {
          throw py::type_error(
              "A call cannot mix buffer and torch tensor inputs");
        }
        saw_torch = true;
#ifdef USE_ATEN_LIB
        auto at_tensor = python_input.cast<at::Tensor>();
        std::vector<int> tensor_sizes(
            at_tensor.sizes().begin(), at_tensor.sizes().end());
        std::vector<int> tensor_strides(
            at_tensor.strides().begin(), at_tensor.strides().end());
        validate_tensor_input(
            method_->method_meta(),
            i,
            runtime_scalar_type(at_tensor),
            tensor_sizes,
            tensor_strides);
        (void)mutable_tensor_data_ptr_no_cow(at_tensor);
        cpp_inputs.emplace_back(at_tensor);
#else
        torch_inputs.push_back(std::make_shared<TorchTensorView>(python_input));
        const auto& view = torch_inputs.back();
        validate_tensor_input(
            method_->method_meta(),
            i,
            view->scalar_type(),
            view->sizes(),
            view->strides());
        auto tensor = for_blob(view->data(), view->sizes(), view->scalar_type())
                          .strides(view->strides())
                          .dim_order(view->dim_order())
                          .dynamism(aten::TensorShapeDynamism::STATIC)
                          .make_tensor_ptr();
        input_tensors.push_back(std::move(tensor));
        cpp_inputs.emplace_back(input_tensors.back());
#endif
      } else if (py::isinstance<py::none>(python_input)) {
        cpp_inputs.push_back(EValue());
      } else if (py::isinstance<py::bool_>(python_input)) {
        cpp_inputs.push_back(EValue(py::cast<bool>(python_input)));
      } else if (py::isinstance<py::int_>(python_input)) {
        cpp_inputs.push_back(EValue(py::cast<int64_t>(python_input)));
      } else if (py::isinstance<py::float_>(python_input)) {
        cpp_inputs.push_back(EValue(py::cast<double>(python_input)));
      } else if (PyObject_CheckBuffer(python_input.ptr())) {
        if (saw_torch) {
          throw py::type_error(
              "A call cannot mix buffer and torch tensor inputs");
        }
        saw_buffer = true;
        buffer_inputs.push_back(std::make_shared<BufferTensor>(python_input));
        const auto& buffer = buffer_inputs.back();
        validate_tensor_input(
            method_->method_meta(),
            i,
            buffer->scalar_type(),
            buffer->sizes(),
            buffer->strides());
#ifdef USE_ATEN_LIB
        auto at_tensor = at::from_blob(
            buffer->data(),
            std::vector<int64_t>(
                buffer->sizes().begin(), buffer->sizes().end()),
            std::vector<int64_t>(
                buffer->strides().begin(), buffer->strides().end()),
            at::TensorOptions().dtype(buffer->scalar_type()));
        cpp_inputs.emplace_back(at_tensor);
#else
        auto tensor =
            for_blob(buffer->data(), buffer->sizes(), buffer->scalar_type())
                .strides(buffer->strides())
                .dim_order(buffer->dim_order())
                .dynamism(aten::TensorShapeDynamism::STATIC)
                .make_tensor_ptr();
        buffer_tensor_ptrs.push_back(std::move(tensor));
        cpp_inputs.emplace_back(buffer_tensor_ptrs.back());
#endif
      } else {
        throw std::runtime_error(
            "Unsupported python type " + type_str +
            ". Ensure that inputs are passed as a flat list of tensors.");
      }
    }

    executorch::aten::ArrayRef<EValue> input_evalue_list(
        cpp_inputs.data(), cpp_inputs.size());

    Error set_inputs_status = method_->set_inputs(input_evalue_list);
    if (set_inputs_status != Error::Ok) {
      // set_inputs() installs values one at a time and does not roll back a
      // prefix when a later value is rejected. Retain both generations so any
      // partially installed input continues to point at live memory.
      buffer_inputs_.insert(
          buffer_inputs_.end(), buffer_inputs.begin(), buffer_inputs.end());
      buffer_tensor_ptrs_.insert(
          buffer_tensor_ptrs_.end(),
          buffer_tensor_ptrs.begin(),
          buffer_tensor_ptrs.end());
#ifndef USE_ATEN_LIB
      torch_inputs_.insert(
          torch_inputs_.end(), torch_inputs.begin(), torch_inputs.end());
      torch_tensor_ptrs_.insert(
          torch_tensor_ptrs_.end(), input_tensors.begin(), input_tensors.end());
#endif
      THROW_IF_ERROR(
          set_inputs_status,
          "method->set_inputs() for method '%s' failed with error 0x%" PRIx32,
          method_->method_meta().name(),
          static_cast<uint32_t>(set_inputs_status));
    }
    buffer_inputs_ = std::move(buffer_inputs);
    buffer_tensor_ptrs_ = std::move(buffer_tensor_ptrs);
#ifndef USE_ATEN_LIB
    torch_inputs_ = std::move(torch_inputs);
    torch_tensor_ptrs_ = std::move(input_tensors);
#endif
  }

  void execute() {
    const auto num_outputs = method_->outputs_size();
    allocate_output_storages();
    std::vector<Span<uint8_t>> output_storage_spans(num_outputs);
    for (int i = 0; i < output_storages_.size(); ++i) {
      output_storage_spans[i] =
          Span<uint8_t>(output_storages_[i].data(), output_storages_[i].size());
    }
#ifdef USE_ATEN_LIB
    // [TLS handling] This is to workaround an assertion failure
    // (https://fburl.com/code/302jyn8d) running `gelu` in ATen mode in fbcode
    // (such as bento). The problem is ExecuTorch ATen mode doesn't have
    // Thread Local State, but `torch-cpp` is assuming tls init is done. There
    // are two more checks: MKLDNN disabled and C10_MOBILE, if any of them is
    // true we won't be hitting this assertion error. However in `torch-cpp`
    // lib both checks are false. Production impact: this should not make any
    // impact in production environment, given that in xplat we are depending
    // on a library that enables C10_MOBILE (`torch_mobile_core`).
    c10::impl::ExcludeDispatchKeyGuard no_autograd(
        c10::autograd_dispatch_keyset);
#endif
    setup_output_storage(*method_, output_storage_spans);
    Error execute_status = method_->execute();
    THROW_IF_ERROR(
        execute_status,
        "method->execute() failed with error 0x%" PRIx32,
        static_cast<uint32_t>(execute_status));
  }

  py::list get_outputs(bool clone_outputs = true) {
    std::vector<EValue> result(method_->outputs_size());

    Error get_outputs_status =
        method_->get_outputs(result.data(), method_->outputs_size());
    THROW_IF_ERROR(
        get_outputs_status,
        "method->get_outputs() for method '%s' failed with error 0x%" PRIx32,
        method_->method_meta().name(),
        static_cast<uint32_t>(get_outputs_status));

    // Retrieve outputs
    return get_outputs_as_py_list(result, clone_outputs);
  }

  py::list call(const py::object& inputs, bool clone_outputs = true) {
    set_inputs(inputs);
    execute();
    return get_outputs(clone_outputs);
  }

  py::object get_attribute(const std::string& name) {
    Result<executorch::aten::Tensor> attr = method_->get_attribute(name);
    THROW_IF_ERROR(
        attr.error(),
        "Failed to get attribute '%s' for method '%s', error: 0x:%" PRIx32,
        name.c_str(),
        method_->method_meta().name(),
        static_cast<uint32_t>(attr.error()));
#ifdef USE_ATEN_LIB
    return py::cast(attr.get());
#else
    return portable_tensor_result(attr.get());
#endif
  }

  PyMethodMeta method_meta() {
    return PyMethodMeta(state_, method_->method_meta());
  }

 private:
  // Method keeps a reference to the memory manager, so we need to keep this
  // alive
  std::shared_ptr<ProgramMemory> memory_;
  // Method keeps a reference to the program, so we also need to keep this alive
  std::shared_ptr<ProgramState> state_;
  std::unique_ptr<Method> method_;
  // Keep Python buffer exports and their TensorPtr metadata alive until the
  // next successful set_inputs() call.
  std::vector<std::shared_ptr<BufferTensor>> buffer_inputs_;
  std::vector<TensorPtr> buffer_tensor_ptrs_;
#ifndef USE_ATEN_LIB
  std::vector<std::shared_ptr<TorchTensorView>> torch_inputs_;
  std::vector<TensorPtr> torch_tensor_ptrs_;
#endif
  // Need to keep-alive output storages until they can be compared in case of
  // bundled programs.
  std::vector<std::vector<uint8_t>> output_storages_;

  void allocate_output_storages() {
    const auto num_outputs = method_->outputs_size();
    // Skip if we already have the right number of storages.
    if (output_storages_.size() == num_outputs) {
      return;
    }
    // Create a buffer for each output tensor. Memory planned outputs and non
    // tensor outputs get an empty buffer in this list which is ignored later.
    output_storages_.reserve(num_outputs);
    auto meta = method_->method_meta();
    for (size_t i = 0; i < num_outputs; ++i) {
      auto output_type = meta.output_tag(i);
      THROW_IF_ERROR(
          output_type.error(), "Failed to get output type for output %zu", i);
      if (output_type.get() != Tag::Tensor) {
        // Skip allocating storage for non-tensor outputs.
        output_storages_.emplace_back();
        continue;
      }
      const auto& output_tensor_meta =
          method_->method_meta().output_tensor_meta(i);
      THROW_IF_ERROR(
          output_tensor_meta.error(),
          "Failed to get output tensor meta for output %zu",
          i);
      if (output_tensor_meta.get().is_memory_planned()) {
        // Skip allocating storage for planned memory outputs.
        output_storages_.emplace_back();
        continue;
      }
      // Allocate storage for the output tensor.
      const size_t output_size = output_tensor_meta.get().nbytes();
      output_storages_.emplace_back(output_size);
    }
  }

  py::list get_outputs_as_py_list(
      const std::vector<EValue>& outputs,
      bool clone_outputs = true) {
    const auto outputs_size = outputs.size();
    py::list list(outputs_size);
    for (size_t i = 0; i < outputs_size; ++i) {
      auto& v = outputs[i];
      if (Tag::None == v.tag) {
        list[i] = py::none();
      } else if (Tag::Int == v.tag) {
        list[i] = py::cast(v.toInt());
      } else if (Tag::Double == v.tag) {
        list[i] = py::cast(v.toDouble());
      } else if (Tag::Bool == v.tag) {
        list[i] = py::cast(v.toBool());
      } else if (Tag::String == v.tag) {
        list[i] = py::cast(std::string(v.toString().data()));
      } else if (Tag::Tensor == v.tag) {
#ifdef USE_ATEN_LIB
        // Clone so the outputs in python do not share a lifetime with the
        // module object
        if (clone_outputs) {
          list[i] = py::cast(v.toTensor().clone());
        } else {
          list[i] = py::cast(v.toTensor());
        }
#else
        (void)clone_outputs;
        list[i] = portable_tensor_result(v.toTensor());
#endif
      } else {
        ET_ASSERT_UNREACHABLE_MSG("Invalid model output type");
      }
    }
    return list;
  }
};

struct PyProgram final {
  explicit PyProgram(
      std::unique_ptr<DataLoader> loader,
      std::unique_ptr<ETDumpGen> tracer = nullptr,
      size_t debug_buffer_size = 0,
      Program::Verification program_verification =
          Program::Verification::Minimal,
      std::optional<const std::string> data_path = std::nullopt)
      : state_(
            load_program(std::move(loader), program_verification, data_path)),
        event_tracer_(std::move(tracer)),
        debug_buffer_size_(debug_buffer_size) {
    // Figure out the size of each non_const layer we need to support every
    // method in the program. Map will be easier to use than a list because we
    // dont know how many non_const arenas there will be
    std::map<size_t, int64_t> non_const_buffer_sizes;
    for (size_t i = 0; i < state_->program_->num_methods(); ++i) {
      auto name = state_->program_->get_method_name(i).get();
      auto method_meta = state_->program_->method_meta(name).get();
      // A device-planned method gets its own arenas in load_method and never
      // reads these, so letting its sizes in would only grow the host arenas
      // the other methods share.
      if (has_device_buffers(method_meta)) {
        continue;
      }
      for (size_t j = 0; j < method_meta.num_memory_planned_buffers(); ++j) {
        auto size = method_meta.memory_planned_buffer_size(j);
        THROW_IF_ERROR(
            size.error(), "Failed to get size of planned buffer %zu", j);
        int64_t buffer_size = size.get();
        if (non_const_buffer_sizes.find(j) == non_const_buffer_sizes.end()) {
          non_const_buffer_sizes.insert({j, buffer_size});
        } else {
          non_const_buffer_sizes[j] =
              std::max(non_const_buffer_sizes[j], buffer_size);
        }
      }
    }

    // Allocate the shared host arenas.
    std::vector<int64_t> planned_sizes;
    planned_sizes.reserve(non_const_buffer_sizes.size());
    for (const auto& entry : non_const_buffer_sizes) {
      planned_sizes.push_back(entry.second);
    }

    memory_ = std::make_shared<ProgramMemory>(std::move(planned_sizes));
    if (event_tracer_ && debug_buffer_size > 0) {
      // If a debug buffer was requested for the ETDump, allocate it and make
      // sure its lifetime is as long as the event_tracer.
      debug_buffer_ = std::make_unique<uint8_t[]>(debug_buffer_size);
      event_tracer_->set_debug_buffer(get_etdump_debug_buffer());
      event_tracer_->set_event_tracer_debug_level(
          EventTracerDebugLogLevel::kIntermediateOutputs);
    }
  }

  static std::unique_ptr<PyProgram> load_from_buffer(
      const py::bytes& buffer,
      bool enable_etdump,
      size_t debug_buffer_size,
      Program::Verification program_verification =
          Program::Verification::Minimal,
      std::optional<const std::string> data_path = std::nullopt) {
    std::unique_ptr<DataLoader> loader = loader_from_buffer(
        buffer.cast<std::string_view>().data(), py::len(buffer));
    return std::make_unique<PyProgram>(
        std::move(loader),
        enable_etdump ? std::make_unique<torch::executor::ETDumpGen>()
                      : nullptr,
        debug_buffer_size,
        program_verification,
        data_path);
  }

  static std::unique_ptr<PyProgram> load_from_file(
      const std::string& path,
      bool enable_etdump,
      size_t debug_buffer_size,
      Program::Verification program_verification =
          Program::Verification::Minimal,
      std::optional<const std::string> data_path = std::nullopt) {
    std::unique_ptr<DataLoader> loader = loader_from_file(path);
    return std::make_unique<PyProgram>(
        std::move(loader),
        enable_etdump ? std::make_unique<torch::executor::ETDumpGen>()
                      : nullptr,
        debug_buffer_size,
        program_verification,
        data_path);
  }

  PyProgram(const PyProgram&) = delete;
  PyProgram& operator=(const PyProgram&) = delete;
  PyProgram(PyProgram&&) = default;
  PyProgram& operator=(PyProgram&&) = default;

  size_t num_methods() const {
    return state_->program_->num_methods();
  }

  std::string get_method_name(size_t method_index) const {
    Result<const char*> res = state_->program_->get_method_name(method_index);
    THROW_IF_ERROR(
        res.error(),
        "Failed get method name, error: 0x:%" PRIx32,
        static_cast<uint32_t>(res.error()));
    return std::string(res.get());
  }

  std::unique_ptr<PyMethod> load_method(const std::string& method_name) {
    Result<MethodMeta> meta =
        state_->program_->method_meta(method_name.c_str());
    THROW_IF_ERROR(
        meta.error(),
        "Failed to get method meta for method %s, error: 0x:%" PRIx32,
        method_name.c_str(),
        static_cast<uint32_t>(meta.error()));
    // Device memory is claimed here rather than at program load so that one
    // accelerator method cannot make the rest of the program unloadable. A
    // host-only method keeps sharing the program-wide arenas, so its planned
    // memory is not isolated from the other host-only methods of this program.
    auto method_memory = make_method_memory(meta.get());
    auto memory = method_memory ? std::move(method_memory) : memory_;
    Result<Method> res = state_->program_->load_method(
        method_name.c_str(),
        memory->mem_manager(),
        event_tracer_.get(),
        state_->data_map_.get());
    THROW_IF_ERROR(
        res.error(),
        "Failed to load method %s, error: 0x:%" PRIx32,
        method_name.c_str(),
        static_cast<uint32_t>(res.error()));
    return std::make_unique<PyMethod>(
        std::move(memory),
        state_,
        std::make_unique<Method>(std::move(res.get())));
  }

  Span<uint8_t> get_etdump_debug_buffer() {
    return Span<uint8_t>(debug_buffer_.get(), debug_buffer_size_);
  }

  std::unique_ptr<PyMethodMeta> method_meta(const std::string& method_name) {
    Result<torch::executor::MethodMeta> res =
        state_->program_->method_meta(method_name.c_str());
    THROW_IF_ERROR(
        res.error(),
        "Failed to get method meta for method %s, error: 0x:%" PRIx32,
        method_name.c_str(),
        static_cast<uint32_t>(res.error()));
    return std::make_unique<PyMethodMeta>(state_, std::move(res.get()));
  }

  bool has_etdump() {
    return static_cast<bool>(event_tracer_);
  }

  void write_etdump_result_to_file(
      const std::string& path,
      const py::object& debug_buffer_path) {
    if (!has_etdump()) {
      throw std::runtime_error("No etdump found");
    }
    auto& etdump = *event_tracer_;
    etdump_result result = etdump.get_etdump_data();
    if (result.buf != nullptr && result.size > 0) {
      write_data_to_file(path, result.buf, result.size);
      free(result.buf);
      if (debug_buffer_size_ > 0 &&
          py::isinstance<py::str>(debug_buffer_path)) {
        // Also write out the debug buffer to a separate file if requested.
        std::string debug_buffer_path_str =
            py::cast<std::string>(debug_buffer_path);
        const auto debug_buffer = get_etdump_debug_buffer();
        write_data_to_file(
            debug_buffer_path_str, debug_buffer.data(), debug_buffer.size());
      }
    } else {
      ET_LOG(
          Info,
          "No etdump data found, try rebuilding with "
          "the CMake option EXECUTORCH_ENABLE_EVENT_TRACER set to ON or with "
          "buck run --config executorch.event_tracer_enabled=true");
    }
  }

 private:
  std::shared_ptr<ProgramMemory> memory_;
  std::shared_ptr<ProgramState> state_;
  std::unique_ptr<ETDumpGen> event_tracer_;
  std::unique_ptr<uint8_t[]> debug_buffer_;
  size_t debug_buffer_size_;
};

void create_profile_block(const std::string& name) {
  EXECUTORCH_PROFILE_CREATE_BLOCK(name.c_str());
}

py::list get_operator_names() {
  Span<const Kernel> kernels = get_registered_kernels();
  py::list res;
  for (const Kernel& k : kernels) {
    if (k.name_ != nullptr) {
      res.append(py::cast(k.name_));
    }
  }
  return res;
}

py::list get_registered_backend_names() {
  size_t n_of_registered_backends = get_num_registered_backends();
  py::list res;
  for (size_t i = 0; i < n_of_registered_backends; i++) {
    auto backend_name_res = get_backend_name(i);
    THROW_IF_ERROR(backend_name_res.error(), "Failed to get backend name");
    auto backend_name = backend_name_res.get();
    res.append(backend_name);
  }
  return res;
}

py::bool_ is_available(const std::string& backend_name) {
  BackendInterface* backend = get_backend_class(backend_name.c_str());
  if (backend == nullptr) {
    return false;
  }
  return backend->is_available();
}

} // namespace

PYBIND11_MODULE(EXECUTORCH_PYTHON_MODULE_NAME, m) {
  // Redirects cout and cerr for function calls this guards to the python env.
  auto call_guard = py::
      call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>();

#ifdef USE_ATEN_LIB
  m.attr("_uses_aten") = true;
#else
  configure_portable_tensor_output();
  m.attr("_uses_aten") = false;
  m.attr("_tensor_output") =
      portable_tensor_output() == PortableTensorOutput::Torch ? "torch"
                                                              : "executorch";
#endif
  py::class_<PyExecuTorchResult, std::shared_ptr<PyExecuTorchResult>>(
      m, "ExecuTorchResult", py::buffer_protocol())
      .def_property_readonly("shape", &PyExecuTorchResult::shape)
      .def_property_readonly("strides", &PyExecuTorchResult::strides)
      .def_property_readonly("dtype", &PyExecuTorchResult::dtype)
      .def_property_readonly("nbytes", &PyExecuTorchResult::nbytes)
      .def_buffer(&PyExecuTorchResult::buffer);

  // Bind the verification enum to python.
  py::enum_<Program::Verification>(m, "Verification")
      .value("Minimal", Program::Verification::Minimal)
      .value("InternalConsistency", Program::Verification::InternalConsistency);

  m.def(
      "_load_for_executorch",
      PyModule::load_from_file,
      py::arg("program_path"),
      py::arg("data_path") = std::nullopt,
      py::arg("enable_etdump") = false,
      py::arg("debug_buffer_size") = 0,
      py::arg("program_verification") =
          Program::Verification::InternalConsistency,
      call_guard);
  m.def(
      "_load_for_executorch_from_buffer",
      &PyModule::load_from_buffer,
      py::arg("buffer"),
      py::arg("data_map_buffer") = std::nullopt,
      py::arg("enable_etdump") = false,
      py::arg("debug_buffer_size") = 0,
      py::arg("program_verification") =
          Program::Verification::InternalConsistency,
      call_guard);
  m.def(
      "_load_for_executorch_from_bundled_program",
      py::overload_cast<
          PyBundledModule&,
          std::optional<const py::bytes>,
          bool,
          size_t>(&PyModule::load_from_bundled_program),
      py::arg("ptr"),
      py::arg("data_map_buffer") = std::nullopt,
      py::arg("enable_etdump") = false,
      py::arg("debug_buffer_size") = 0,
      call_guard);
  m.def(
      "_load_for_executorch_from_bundled_program",
      py::overload_cast<PyBundledModule&, const std::string&, bool, size_t>(
          &PyModule::load_from_bundled_program),
      py::arg("ptr"),
      py::arg("data_path"),
      py::arg("enable_etdump") = false,
      py::arg("debug_buffer_size") = 0,
      call_guard);
  m.def(
      "_load_bundled_program_from_buffer",
      &PyBundledModule::load_from_buffer,
      py::arg("buffer"),
      py::arg("non_const_pool_size") = kDEFAULT_BUNDLED_INPUT_POOL_SIZE,
      call_guard);

  // Import the PyDataLoader type from the shared module.
  // This ensures the type is registered once and shared across all modules.
#ifdef USE_ATEN_LIB
  py::module_::import("executorch.extension.pybindings.data_loader");
#else
  // A standalone torch-free extension can be imported directly for embedded
  // use. The data-loader overload remains unavailable until its shared type is
  // installed, but the rest of the runtime does not require that package.
  try {
    py::module_::import("executorch.extension.pybindings.data_loader");
  } catch (py::error_already_set& error) {
    if (!error.matches(PyExc_ModuleNotFoundError) ||
        py::str(error.value().attr("name")).cast<std::string>() !=
            "executorch") {
      throw;
    }
    error.clear();
  }
#endif

  m.def(
      "_load_for_executorch_from_data_loader",
      &PyModule::load_from_data_loader,
      py::arg("loader"),
      py::arg("data_path") = py::none(),
      py::arg("enable_etdump") = false,
      py::arg("debug_buffer_size") = 0,
      call_guard);

  m.def(
      "_dump_profile_results",
      []() {
        prof_result_t prof_result;
        EXECUTORCH_DUMP_PROFILE_RESULTS(&prof_result);
        return py::bytes(
            reinterpret_cast<const char*>(prof_result.prof_data),
            prof_result.num_bytes);
      },
      call_guard);
  m.def(
      "_get_registered_backend_names",
      &get_registered_backend_names,
      call_guard);
  m.def("_get_operator_names", &get_operator_names);
  m.def("_is_available", &is_available, py::arg("backend_name"), call_guard);
  m.def("_create_profile_block", &create_profile_block, call_guard);
  m.def(
      "_reset_profile_results",
      []() { EXECUTORCH_RESET_PROFILE_RESULTS(); },
      call_guard);
  m.def(
      "_unsafe_reset_threadpool",
      [](int num_threads) {
        executorch::extension::threadpool::get_threadpool()
            ->_unsafe_reset_threadpool(num_threads);
      },
      py::arg("num_threads"),
      call_guard);
  m.def(
      "_threadpool_get_thread_count",
      []() {
        return ::executorch::extension::threadpool::get_threadpool()
            ->get_thread_count();
      },
      call_guard);

  py::class_<PyModule>(m, "ExecuTorchModule")
      .def(
          "plan_execute",
          &PyModule::plan_execute,
          py::arg("method_name"),
          py::arg("clone_outputs") = true,
          call_guard)
      .def(
          "method_meta",
          &PyModule::method_meta,
          py::arg("method_name"),
          call_guard)
      .def("method_names", &PyModule::method_names, call_guard)
      .def(
          "run_method",
          &PyModule::run_method,
          py::arg("method_name"),
          py::arg("inputs") = py::list(),
          py::arg("clone_outputs") = true,
          call_guard)
      .def(
          "forward",
          &PyModule::forward,
          py::arg("inputs") = py::list(),
          py::arg("clone_outputs") = true,
          call_guard)
      .def("has_etdump", &PyModule::has_etdump, call_guard)
      .def(
          "write_etdump_result_to_file",
          &PyModule::write_etdump_result_to_file,
          py::arg("path"),
          py::arg("debug_buffer_path") = py::none(),
          call_guard)
      .def(
          "__call__",
          &PyModule::forward,
          py::arg("inputs") = py::list(),
          py::arg("clone_outputs") = true,
          call_guard);

  py::class_<PyBundledModule>(m, "BundledModule")
      .def(
          "verify_result_with_bundled_expected_output",
          &PyBundledModule::verify_result_with_bundled_expected_output,
          py::arg("method_name"),
          py::arg("testset_idx"),
          py::arg("rtol") = 1e-5,
          py::arg("atol") = 1e-8,
          call_guard);

  py::class_<PyTensorInfo>(m, "TensorInfo")
      .def("sizes", &PyTensorInfo::sizes, call_guard)
      .def("dtype", &PyTensorInfo::dtype, call_guard)
      .def("is_memory_planned", &PyTensorInfo::is_memory_planned, call_guard)
      .def("nbytes", &PyTensorInfo::nbytes, call_guard)
      .def("__repr__", &PyTensorInfo::repr, call_guard);
  py::class_<PyMethodMeta>(m, "MethodMeta")
      .def("name", &PyMethodMeta::name, call_guard)
      .def("num_inputs", &PyMethodMeta::num_inputs, call_guard)
      .def("num_outputs", &PyMethodMeta::num_outputs, call_guard)
      .def("num_attributes", &PyMethodMeta::num_attributes, call_guard)
      .def(
          "input_tensor_meta",
          &PyMethodMeta::input_tensor_meta,
          py::arg("index"),
          call_guard)
      .def(
          "output_tensor_meta",
          &PyMethodMeta::output_tensor_meta,
          py::arg("index"),
          call_guard)
      .def(
          "attribute_tensor_meta",
          &PyMethodMeta::attribute_tensor_meta,
          py::arg("index"),
          call_guard)
      .def("__repr__", &PyMethodMeta::repr, call_guard);

  m.def(
      "_load_program",
      &PyProgram::load_from_file,
      py::arg("path"),
      py::arg("enable_etdump") = false,
      py::arg("debug_buffer_size") = 0,
      py::arg("program_verification") = Program::Verification::Minimal,
      py::arg("data_path") = std::nullopt,
      call_guard);
  m.def(
      "_load_program_from_buffer",
      &PyProgram::load_from_buffer,
      py::arg("buffer"),
      py::arg("enable_etdump") = false,
      py::arg("debug_buffer_size") = 0,
      py::arg("program_verification") = Program::Verification::Minimal,
      py::arg("data_path") = std::nullopt,
      call_guard);
  py::class_<PyProgram>(m, "ExecuTorchProgram")
      .def("num_methods", &PyProgram::num_methods, call_guard)
      .def(
          "get_method_name",
          &PyProgram::get_method_name,
          py::arg("method_index"),
          call_guard)
      .def(
          "load_method",
          &PyProgram::load_method,
          py::arg("method_name"),
          call_guard)
      .def(
          "method_meta",
          &PyProgram::method_meta,
          py::arg("method_name"),
          call_guard)
      .def("has_etdump", &PyProgram::has_etdump, call_guard)
      .def(
          "write_etdump_result_to_file",
          &PyProgram::write_etdump_result_to_file,
          py::arg("path"),
          py::arg("debug_buffer_path") = py::none(),
          call_guard);
  py::class_<PyMethod>(m, "ExecuTorchMethod")
      .def("set_inputs", &PyMethod::set_inputs, py::arg("inputs"), call_guard)
      .def("execute", &PyMethod::execute, call_guard)
      .def(
          "get_outputs",
          &PyMethod::get_outputs,
          py::arg("clone_outputs") = true,
          call_guard)
      .def(
          "call",
          &PyMethod::call,
          py::arg("inputs") = py::list(),
          py::arg("clone_outputs") = true,
          call_guard)
      .def(
          "__call__",
          &PyMethod::call,
          py::arg("inputs") = py::list(),
          py::arg("clone_outputs") = true,
          call_guard)
      .def(
          "get_attribute",
          &PyMethod::get_attribute,
          py::arg("name"),
          call_guard)
      .def("method_meta", &PyMethod::method_meta, call_guard);
}

namespace {

// Our logs work by writing to stderr. By default this is done through fprintf
// (as defined in posix.cpp) which then does not show up in python environments.
// Here we override the pal to use std::cerr which can be properly redirected by
// scoped_estream_redirect.
void emit_log_message(
    et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    ET_UNUSED const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  std::cerr << "[" << filename << ":" << line << "] " << message << std::endl;
}

runtime::PalImpl build_pal() {
  return runtime::PalImpl::create(emit_log_message, __FILE__);
}

// Update PAL to redirect logs.
ET_UNUSED bool registration_result = runtime::register_pal(build_pal());

} // namespace

} // namespace pybindings
} // namespace extension
} // namespace executorch
