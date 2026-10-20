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
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

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
#include <executorch/extension/pybindings/pybindings_cuda.h>
#include <executorch/extension/pybindings/pybindings_data_loader.h>
#include <executorch/extension/pybindings/pybindings_dlpack.h>
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
using ::executorch::runtime::BackendOption;
using ::executorch::runtime::DataLoader;
using ::executorch::runtime::DeviceMemoryBuffer;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::EventTracerDebugLogLevel;
using ::executorch::runtime::HierarchicalAllocator;
using ::executorch::runtime::kMaxOptionKeyLength;
using ::executorch::runtime::kMaxOptionValueLength;
using ::executorch::runtime::LoadBackendOptionsMap;
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
  if (value != nullptr && std::strcmp(value, "executorch") == 0) {
    portable_tensor_output() = PortableTensorOutput::ExecuTorch;
    return;
  }
  if (value != nullptr && std::strcmp(value, "torch") != 0) {
    throw py::value_error(
        std::string(kTensorOutputEnvironmentVariable) +
        " must be either 'torch' or 'executorch', but was '" + value + "'");
  }
  try {
    py::module_::import("torch");
    (void)loaded_torch_module();
    portable_tensor_output() = PortableTensorOutput::Torch;
  } catch (py::error_already_set& error) {
    if (value != nullptr || !error.matches(PyExc_ImportError)) {
      throw;
    }
    portable_tensor_output() = PortableTensorOutput::ExecuTorch;
  }
}
#endif

// Python tensor adapters can release the GIL while holding native state.
// Recursive locking also covers compound calls that invoke other methods.
class ObjectLock final {
 public:
  explicit ObjectLock(std::recursive_mutex& mutex)
      : lock_(mutex, std::defer_lock) {
    py::gil_scoped_release release;
    lock_.lock();
  }

 private:
  std::unique_lock<std::recursive_mutex> lock_;
};

#ifdef USE_ATEN_LIB
void* mutable_tensor_data_ptr_no_cow(const at::Tensor& tensor) {
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

// Copies dense data into the strides of another dimension order. The source
// strides must describe a dense layout of `sizes`.
template <size_t kItemSize>
void copy_to_layout(
    const uint8_t* from,
    const std::vector<int>& source_strides,
    uint8_t* to,
    const std::vector<int>& destination_strides,
    const std::vector<int>& sizes) {
  const size_t rank = sizes.size();
  size_t count = 1;
  for (const auto size : sizes) {
    count *= static_cast<size_t>(size);
  }
  std::vector<int> index(rank, 0);
  ptrdiff_t from_offset = 0;
  ptrdiff_t to_offset = 0;
  for (size_t element = 0; element < count; ++element) {
    std::memcpy(
        to + to_offset * static_cast<ptrdiff_t>(kItemSize),
        from + from_offset * static_cast<ptrdiff_t>(kItemSize),
        kItemSize);
    for (size_t dim = rank; dim > 0; --dim) {
      from_offset += source_strides[dim - 1];
      to_offset += destination_strides[dim - 1];
      if (++index[dim - 1] < sizes[dim - 1]) {
        break;
      }
      from_offset -=
          static_cast<ptrdiff_t>(sizes[dim - 1]) * source_strides[dim - 1];
      to_offset -=
          static_cast<ptrdiff_t>(sizes[dim - 1]) * destination_strides[dim - 1];
      index[dim - 1] = 0;
    }
  }
}

void copy_to_layout(
    const void* source,
    const std::vector<int>& source_strides,
    void* destination,
    const std::vector<int>& destination_strides,
    const std::vector<int>& sizes,
    size_t itemsize) {
  const auto* from = static_cast<const uint8_t*>(source);
  auto* to = static_cast<uint8_t*>(destination);
  switch (itemsize) {
    case 1:
      return copy_to_layout<1>(
          from, source_strides, to, destination_strides, sizes);
    case 2:
      return copy_to_layout<2>(
          from, source_strides, to, destination_strides, sizes);
    case 4:
      return copy_to_layout<4>(
          from, source_strides, to, destination_strides, sizes);
    case 8:
      return copy_to_layout<8>(
          from, source_strides, to, destination_strides, sizes);
    case 16:
      return copy_to_layout<16>(
          from, source_strides, to, destination_strides, sizes);
    default:
      throw std::runtime_error(
          "Unsupported element size " + std::to_string(itemsize));
  }
}

// Whether data dense in `strides` is already laid out in `dim_order`. Size-one
// dimensions can sit anywhere in either order without moving any data.
bool has_layout(
    const std::vector<int>& sizes,
    const std::vector<int>& strides,
    Span<const uint8_t> dim_order) {
  std::vector<int> expected(sizes.size());
  if (runtime::dim_order_to_stride(
          sizes.data(), dim_order.data(), sizes.size(), expected.data()) !=
      Error::Ok) {
    return false;
  }
  for (size_t dim = 0; dim < sizes.size(); ++dim) {
    if (sizes[dim] > 1 && strides[dim] != expected[dim]) {
      return false;
    }
  }
  return true;
}

/** Holds a Python buffer export and presents its metadata to ExecuTorch. */
class BufferTensor final {
 public:
  BufferTensor(
      const py::handle& value,
      const MethodMeta& method_meta,
      size_t input_index)
      : buffer_(py::reinterpret_borrow<py::buffer>(value)),
        info_(buffer_.request()) {
    check_rank();
    const auto input_meta = method_meta.input_tensor_meta(input_index);
    THROW_IF_ERROR(input_meta.error(), "not a tensor input");
    const bool raw_bfloat16 =
        input_meta->scalar_type() == executorch::aten::ScalarType::BFloat16 &&
        info_.ndim == 1 &&
        scalar_type_from_buffer(info_) == executorch::aten::ScalarType::Byte;
    if (raw_bfloat16) {
      if (info_.strides.size() != 1 || info_.strides[0] != 1) {
        throw py::buffer_error(
            "Raw bfloat16 inputs require contiguous byte buffers");
      }
      if (static_cast<size_t>(info_.size) != input_meta->nbytes()) {
        throw py::value_error(
            "raw bfloat16 input has " + std::to_string(info_.size) +
            " bytes, but the method expects " +
            std::to_string(input_meta->nbytes()));
      }
      scalar_type_ = executorch::aten::ScalarType::BFloat16;
      sizes_.assign(input_meta->sizes().begin(), input_meta->sizes().end());
      dim_order_.assign(
          input_meta->dim_order().begin(), input_meta->dim_order().end());
      strides_.resize(sizes_.size());
      const auto status = runtime::dim_order_to_stride(
          sizes_.data(), dim_order_.data(), sizes_.size(), strides_.data());
      THROW_IF_ERROR(
          status, "Invalid dimension order for input %zu", input_index);
    } else {
      read_layout();
    }
    if ((info_.readonly || raw_bfloat16) && info_.size > 0) {
      owned_data_.resize(info_.size * info_.itemsize);
      std::memcpy(owned_data_.data(), info_.ptr, owned_data_.size());
      info_ = py::buffer_info(
          owned_data_.data(),
          info_.itemsize,
          info_.format,
          info_.ndim,
          info_.shape,
          info_.strides);
      buffer_ = py::buffer();
    }
  }

  // A buffer that is not bound to a method input. Its data is borrowed, not
  // copied.
  explicit BufferTensor(const py::handle& value)
      : buffer_(py::reinterpret_borrow<py::buffer>(value)),
        info_(buffer_.request()) {
    check_rank();
    read_layout();
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

  // Describes the data in the exported dimension order. Data dense in another
  // order is copied into that order, since reading it with the exported
  // strides would permute its values. Empty data needs no copy. ATen builds
  // keep the data as it is: there the runtime gives each input contiguous
  // strides when it resizes it, so only contiguous data reads correctly.
  void use_layout(Span<const uint8_t> dim_order) {
    // The input check that follows reports a rank mismatch.
    if (dim_order.size() != sizes_.size()) {
      return;
    }
    const bool empty = has_empty_dimension();
#ifdef USE_ATEN_LIB
    if (!empty) {
      return;
    }
#endif
    if (!empty && has_layout(sizes_, strides_, dim_order)) {
      return;
    }
    std::vector<int> strides(sizes_.size());
    const auto status = runtime::dim_order_to_stride(
        sizes_.data(), dim_order.data(), sizes_.size(), strides.data());
    THROW_IF_ERROR(status, "Invalid dimension order for buffer input");
    if (!empty) {
      std::vector<uint8_t> converted(nbytes());
      copy_to_layout(
          data(),
          strides_,
          converted.data(),
          strides,
          sizes_,
          static_cast<size_t>(info_.itemsize));
      owned_data_ = std::move(converted);
      std::vector<py::ssize_t> byte_strides(strides.begin(), strides.end());
      for (auto& stride : byte_strides) {
        stride *= info_.itemsize;
      }
      // Drop the export too, so the caller's object is released.
      info_ = py::buffer_info(
          owned_data_.data(),
          info_.itemsize,
          info_.format,
          info_.ndim,
          info_.shape,
          byte_strides);
      buffer_ = py::buffer();
    }
    strides_ = std::move(strides);
    dim_order_.assign(dim_order.begin(), dim_order.end());
  }

  bool borrows_data() const {
    return owned_data_.empty() && info_.size > 0;
  }

  std::pair<const void*, size_t> current_storage() const {
    const auto current = buffer_.request();
    return {
        current.ptr,
        static_cast<size_t>(current.size) *
            static_cast<size_t>(current.itemsize)};
  }

  size_t nbytes() const {
    return static_cast<size_t>(info_.size) *
        static_cast<size_t>(info_.itemsize);
  }

  py::object result_backing_owner() const {
    if (!owned_data_.empty()) {
      return py::none();
    }
    PyObject* numpy = PyDict_GetItemString(PyImport_GetModuleDict(), "numpy");
    py::object array_type = py::none();
    if (numpy != nullptr) {
      array_type = py::reinterpret_borrow<py::object>(numpy).attr("ndarray");
    }
    py::object source = buffer_;
    while (source &&
           ((!array_type.is_none() && py::isinstance(source, array_type)) ||
            py::isinstance<py::memoryview>(source))) {
      source = !array_type.is_none() && py::isinstance(source, array_type)
          ? py::reinterpret_borrow<py::array>(source).base()
          : source.attr("obj");
    }
    if (source && py::isinstance<PyExecuTorchResult>(source)) {
      return source.cast<std::shared_ptr<PyExecuTorchResult>>()->backing_owner(
          source);
    }
    return py::none();
  }

 private:
  void check_rank() const {
    if (info_.ndim < 0 ||
        static_cast<size_t>(info_.ndim) > runtime::kTensorDimensionLimit) {
      throw py::buffer_error("Buffer rank is too large for ExecuTorch");
    }
  }

  void read_layout() {
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
    if (!has_empty_dimension()) {
      validate_dense_layout();
    }
  }

  bool has_empty_dimension() const {
    return std::find(sizes_.begin(), sizes_.end(), 0) != sizes_.end();
  }

  static int checked_int(py::ssize_t value, const char* name) {
    if (value < 0 || value > std::numeric_limits<int>::max()) {
      throw py::buffer_error(
          std::string("Buffer ") + name + " is too large for ExecuTorch");
    }
    return static_cast<int>(value);
  }

  // PEP 3118 formats: an optional byte-order prefix, then a struct code. The
  // code gives the kind of number and itemsize gives its width, which also
  // covers platform-sized codes such as 'l' and 'n'. ctypes spells out the
  // native byte order with a prefix.
  static executorch::aten::ScalarType scalar_type_from_buffer(
      const py::buffer_info& info) {
    using executorch::aten::ScalarType;
    const std::string& format = info.format;
    const uint16_t one = 1;
    const bool little_endian = *reinterpret_cast<const uint8_t*>(&one) == 1;
    size_t code_offset = 0;
    if (!format.empty() &&
        std::string("@=<>!").find(format[0]) != std::string::npos) {
      const bool native = format[0] == '@' || format[0] == '=' ||
          (format[0] == '<' && little_endian) ||
          (format[0] != '<' && !little_endian);
      if (!native) {
        throw py::buffer_error(
            "Buffer format '" + format +
            "' is not in this machine's byte order. Convert the data to the "
            "native byte order before passing it.");
      }
      code_offset = 1;
    }
    const auto type_for_itemsize =
        [&info](std::initializer_list<std::pair<int, ScalarType>> types)
        -> std::optional<ScalarType> {
      for (const auto& [itemsize, type] : types) {
        if (info.itemsize == itemsize) {
          return type;
        }
      }
      return std::nullopt;
    };
    std::optional<ScalarType> type;
    if (format.size() == code_offset + 1) {
      switch (format[code_offset]) {
        case '?':
          type = type_for_itemsize({{1, ScalarType::Bool}});
          break;
        case 'b':
        case 'h':
        case 'i':
        case 'l':
        case 'q':
        case 'n':
          type = type_for_itemsize(
              {{1, ScalarType::Char},
               {2, ScalarType::Short},
               {4, ScalarType::Int},
               {8, ScalarType::Long}});
          break;
        case 'B':
        case 'H':
        case 'I':
        case 'L':
        case 'Q':
        case 'N':
          type = type_for_itemsize(
              {{1, ScalarType::Byte},
               {2, ScalarType::UInt16},
               {4, ScalarType::UInt32},
               {8, ScalarType::UInt64}});
          break;
        case 'e':
        case 'f':
        case 'd':
          type = type_for_itemsize(
              {{2, ScalarType::Half},
               {4, ScalarType::Float},
               {8, ScalarType::Double}});
          break;
      }
    } else if (format.size() == code_offset + 2 && format[code_offset] == 'Z') {
      if (format[code_offset + 1] == 'f' || format[code_offset + 1] == 'd') {
        type = type_for_itemsize(
            {{8, ScalarType::ComplexFloat}, {16, ScalarType::ComplexDouble}});
      }
    }
    if (!type) {
      throw py::buffer_error(
          "Unsupported buffer format '" + format + "' with item size " +
          std::to_string(info.itemsize) +
          ". Use a buffer of booleans, integers, floats, or complex numbers.");
    }
    return *type;
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
  static PyObject* tensor_type = nullptr;
  if (tensor_type == nullptr) {
    PyObject* torch_module = loaded_torch_module();
    if (torch_module == nullptr) {
      return false;
    }
    tensor_type = PyObject_GetAttrString(torch_module, "Tensor");
    if (tensor_type == nullptr) {
      throw py::error_already_set();
    }
  }
  const int result = PyObject_IsInstance(value.ptr(), tensor_type);
  if (result < 0) {
    throw py::error_already_set();
  }
  return result == 1;
}

// Runs `read`, which builds one input from a Python object, and names the
// input and its method in its errors, which the readers cannot see. A Python
// error keeps its exact type and gets the name as a note; the C++ errors
// caught below get it in their message. Others, such as std::bad_alloc, pass
// through unnamed, so pybind11 maps them to their own Python type.
template <typename Read>
auto read_input(const MethodMeta& method_meta, size_t index, Read&& read) {
  const auto name = [&] {
    return "Input " + std::to_string(index) + " of method '" +
        method_meta.name() + "'";
  };
  const auto named = [&name](const char* message) {
    return name() + ": " + message;
  };
  try {
    return read();
  } catch (py::error_already_set& error) {
    // Rebuilding the exception as type(message) would fail for any exception
    // whose constructor is not a single string.
    if (error.value()) {
      try {
        error.value().attr("add_note")(name());
      } catch (...) {
        // add_note is best effort; never hide the original error.
      }
    }
    throw;
  } catch (const py::buffer_error& error) {
    throw py::buffer_error(named(error.what()));
  } catch (const py::value_error& error) {
    throw py::value_error(named(error.what()));
  } catch (const py::type_error& error) {
    throw py::type_error(named(error.what()));
  } catch (const std::invalid_argument& error) {
    throw py::value_error(named(error.what()));
  } catch (const std::runtime_error& error) {
    throw std::runtime_error(named(error.what()));
  }
}

void validate_tensor_input(
    const MethodMeta& method_meta,
    size_t index,
    executorch::aten::ScalarType scalar_type,
    const std::vector<int>& sizes) {
  const auto input_meta = method_meta.input_tensor_meta(index);
  THROW_IF_ERROR(
      input_meta.error(),
      "Input %zu of method '%s' is not a tensor input",
      index,
      method_meta.name());
  const std::string input_name = "Input " + std::to_string(index) +
      " of method '" + method_meta.name() + "'";
  if (input_meta->scalar_type() != scalar_type) {
    throw py::value_error(
        input_name + " has dtype " +
        std::string(runtime::toString(scalar_type)) +
        " but the method expects " +
        std::string(runtime::toString(input_meta->scalar_type())));
  }
  const auto expected_rank = input_meta->sizes().size();
  if (expected_rank != sizes.size()) {
    throw py::value_error(
        input_name + " has rank " + std::to_string(sizes.size()) +
        ", but the method expects rank " + std::to_string(expected_rank));
  }
}

void validate_buffer_input(
    const MethodMeta& method_meta,
    size_t index,
    BufferTensor& buffer) {
  const auto input_meta = method_meta.input_tensor_meta(index);
  if (input_meta.ok()) {
    buffer.use_layout(input_meta->dim_order());
  }
  validate_tensor_input(
      method_meta, index, buffer.scalar_type(), buffer.sizes());
}

#ifndef USE_ATEN_LIB
void validate_input_device(
    const Method& method,
    size_t index,
    executorch::aten::Device device,
    size_t nbytes) {
  if (nbytes == 0) {
    return;
  }
  const auto expected = method.get_input(index).toTensor().device();
  if (device != expected) {
    const auto name = [](executorch::aten::Device value) {
      return value.is_cpu() ? std::string("cpu")
                            : "cuda:" + std::to_string(value.index());
    };
    std::string message = "Input " + std::to_string(index) + " of method '" +
        method.method_meta().name() + "' is on device " + name(device) +
        ", but the method expects " + name(expected);
    if (device.is_cpu()) {
      message +=
          ". Pass device memory instead, such as "
          "ExecuTorchResult.from_host(...) or a CUDA tensor";
    }
    throw py::value_error(message);
  }
}

/** A torch.Tensor input read through its public Python API, copied only to
 * change its layout. */
class TorchTensorView final {
 public:
  TorchTensorView(const py::handle& value, size_t index)
      : owner_(py::reinterpret_borrow<py::object>(value)), index_(index) {
    if (owner_.attr("is_neg")().cast<bool>() ||
        owner_.attr("is_conj")().cast<bool>()) {
      throw py::value_error(
          "Lazy conjugate and negative torch tensor views must be resolved before execution");
    }
    const auto numel = owner_.attr("numel")().cast<int64_t>();
    const auto device_type =
        py::str(owner_.attr("device").attr("type")).cast<std::string>();
    if (device_type == "cuda") {
      const auto index = owner_.attr("device").attr("index").cast<int>();
      if (index < 0 ||
          index > std::numeric_limits<runtime::etensor::DeviceIndex>::max()) {
        throw py::value_error(
            "Torch CUDA device index is too large for ExecuTorch");
      }
      device_ =
          executorch::aten::Device(executorch::aten::DeviceType::CUDA, index);
    } else if (device_type != "cpu" && numel != 0) {
      throw std::runtime_error(
          "Torch tensor is on device " + device_type +
          ", and only CPU and CUDA tensors can be passed to portable bindings.");
    }
    scalar_type_ = scalar_type_from_torch_dtype(
        py::str(owner_.attr("dtype")).cast<std::string>());
    nbytes_ = numel * runtime::elementSize(scalar_type_);
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
    if (numel != 0) {
      validate_dense_layout();
    }
    const auto address =
        numel == 0 ? 0 : owner_.attr("data_ptr")().cast<uintptr_t>();
    if (address == 0 && numel != 0) {
      throw py::value_error("Torch tensor data is not allocated");
    }
    data_ = reinterpret_cast<void*>(address);
  }

  size_t index() const {
    return index_;
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

  size_t nbytes() const {
    return nbytes_;
  }

  executorch::aten::Device device() const {
    return device_;
  }

  const py::object& owner() const {
    return owner_;
  }

  // Describes the tensor in the exported dimension order. A tensor dense in
  // another order is first copied by torch into that order on its own device,
  // since reading it with the exported strides would permute its values.
  // owner() is then the copy.
  void use_layout(Span<const uint8_t> dim_order) {
    // The exported strides are computed as int. The input's own strides fit
    // in int already, so only a conversion can need a larger one.
    int64_t outer_stride = 1;
    for (size_t i = 1; i < dim_order.size(); ++i) {
      outer_stride *= std::max(sizes_[dim_order[i]], 1);
    }
    if (outer_stride > std::numeric_limits<int>::max()) {
      throw py::value_error(
          "Torch tensor is too large to convert to the exported layout");
    }
    std::vector<int> strides(sizes_.size());
    const auto status = runtime::dim_order_to_stride(
        sizes_.data(), dim_order.data(), sizes_.size(), strides.data());
    THROW_IF_ERROR(status, "Invalid dimension order for torch tensor input");
    if (nbytes_ != 0 && !has_layout(sizes_, strides_, dim_order)) {
      // torch copies on its current stream, which may run before a write the
      // producer queued on another stream, so wait for the device first.
      if (device_.type() == executorch::aten::DeviceType::CUDA) {
        synchronize_torch_cuda_device(device_.index());
      }
      owner_ = py::reinterpret_borrow<py::object>(loaded_torch_module())
                   .attr("empty_strided")(
                       sizes_,
                       strides,
                       py::arg("dtype") = owner_.attr("dtype"),
                       py::arg("device") = owner_.attr("device"))
                   .attr("copy_")(owner_);
      data_ =
          reinterpret_cast<void*>(owner_.attr("data_ptr")().cast<uintptr_t>());
    }
    strides_ = std::move(strides);
    dim_order_.assign(dim_order.begin(), dim_order.end());
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
  size_t index_;
  void* data_ = nullptr;
  std::vector<int> sizes_;
  std::vector<int> strides_;
  std::vector<uint8_t> dim_order_;
  executorch::aten::ScalarType scalar_type_;
  size_t nbytes_;
  executorch::aten::Device device_{executorch::aten::DeviceType::CPU};
};
#endif

class BorrowedInput final {
 public:
  BorrowedInput(size_t index, const py::handle& torch_tensor)
      : index_(index),
        torch_tensor_(py::reinterpret_borrow<py::object>(torch_tensor)) {
    std::tie(data_, nbytes_) = torch_storage();
  }

  BorrowedInput(size_t index, std::shared_ptr<BufferTensor> buffer)
      : index_(index),
        buffer_(std::move(buffer)),
        data_(buffer_->data()),
        nbytes_(buffer_->nbytes()) {}

  void validate(const char* method_name) const {
    const auto [current_data, current_nbytes] =
        buffer_ ? buffer_->current_storage() : torch_storage();
    if (current_data != data_ || current_nbytes != nbytes_) {
      throw py::value_error(
          "Input " + std::to_string(index_) + " of method '" + method_name +
          "': its storage changed after it was passed in; pass it again");
    }
  }

 private:
  std::pair<const void*, size_t> torch_storage() const {
    const auto numel = torch_tensor_.attr("numel")().cast<size_t>();
    const auto address =
        numel == 0 ? 0 : torch_tensor_.attr("data_ptr")().cast<uintptr_t>();
    const auto element_size =
        torch_tensor_.attr("element_size")().cast<size_t>();
    return {reinterpret_cast<const void*>(address), numel * element_size};
  }

  size_t index_;
  py::object torch_tensor_;
  std::shared_ptr<BufferTensor> buffer_;
  const void* data_;
  size_t nbytes_;
};

#ifndef USE_ATEN_LIB
// Asks a producer for a capsule of CUDA memory on device `index`. Producers
// such as torch export a CUDA tensor only while its device is the current one,
// and torch reads that from its own state, so a torch that can use CUDA is
// asked to switch.
py::object cuda_dlpack_capsule(const py::object& producer, int index) {
  const auto request = [&producer] {
    return producer.attr("__dlpack__")(py::arg("stream") = 1);
  };
  PyObject* torch = loaded_torch_module();
  const auto torch_cuda = torch == nullptr
      ? py::object(py::none())
      : py::reinterpret_borrow<py::object>(torch).attr("cuda");
  if (torch_cuda.is_none() || !torch_cuda.attr("is_available")().cast<bool>()) {
    return with_cuda_device(index, request);
  }
  const auto device = torch_cuda.attr("device")(index);
  device.attr("__enter__")();
  try {
    auto capsule = request();
    device.attr("__exit__")(py::none(), py::none(), py::none());
    return capsule;
  } catch (...) {
    device.attr("__exit__")(py::none(), py::none(), py::none());
    throw;
  }
}

/** A borrowed tensor view obtained through the producer's DLPack protocol. */
class DLPackTensorView final {
 public:
  explicit DLPackTensorView(const py::handle& value)
      : owner_(py::reinterpret_borrow<py::object>(value)),
        advertised_device_(
            owner_.attr("__dlpack_device__")().cast<std::pair<int, int>>()) {
    validate_advertised_device();
    // A result of these bindings gives its capsule without the wait in
    // __dlpack__, since the call waits for every CUDA input before it runs.
    if (py::isinstance<PyExecuTorchResult>(owner_)) {
      capsule_ = owner_.cast<std::shared_ptr<PyExecuTorchResult>>()->capsule();
    } else {
      capsule_ = advertised_device_.first == dlpack::CUDA
          ? cuda_dlpack_capsule(owner_, advertised_device_.second)
          : owner_.attr("__dlpack__")();
    }
    if (!PyCapsule_IsValid(capsule_.ptr(), "dltensor")) {
      throw py::value_error("__dlpack__() did not return a dltensor capsule");
    }
    managed_ = static_cast<dlpack::ManagedTensor*>(
        PyCapsule_GetPointer(capsule_.ptr(), "dltensor"));
    if (managed_ == nullptr) {
      throw py::error_already_set();
    }
    initialize(managed_->dl_tensor);
    if (PyCapsule_SetName(capsule_.ptr(), "used_dltensor") != 0) {
      throw py::error_already_set();
    }
  }

  ~DLPackTensorView() {
    if (managed_ != nullptr && managed_->deleter != nullptr) {
      managed_->deleter(managed_);
    }
  }

  DLPackTensorView(const DLPackTensorView&) = delete;
  DLPackTensorView& operator=(const DLPackTensorView&) = delete;

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

  executorch::aten::Device device() const {
    return device_;
  }

  size_t nbytes() const {
    return nbytes_;
  }

  // Describes the tensor in the exported dimension order. A CPU tensor dense
  // in another order is copied into that order, since reading it with the
  // exported strides would permute its values. A device tensor would need a
  // device copy, so it must already be in the exported order.
  void use_layout(Span<const uint8_t> dim_order) {
    std::vector<int> strides(sizes_.size());
    const auto status = runtime::dim_order_to_stride(
        sizes_.data(), dim_order.data(), sizes_.size(), strides.data());
    THROW_IF_ERROR(status, "Invalid dimension order for DLPack tensor input");
    if (nbytes_ != 0 && !has_layout(sizes_, strides_, dim_order)) {
      if (!device_.is_cpu()) {
        throw py::value_error(
            "DLPack device input is laid out differently from the exported "
            "method. Pass the input in the layout the method was exported "
            "with.");
      }
      converted_.resize(nbytes_);
      copy_to_layout(
          data_,
          strides_,
          converted_.data(),
          strides,
          sizes_,
          runtime::elementSize(scalar_type_));
      data_ = converted_.data();
    }
    strides_ = std::move(strides);
    dim_order_.assign(dim_order.begin(), dim_order.end());
  }

  py::object result_backing_owner() const {
    if (!converted_.empty()) {
      return py::none();
    }
    auto capsule_owner = PyExecuTorchResult::dlpack_backing_owner(managed_);
    if (!capsule_owner.is_none()) {
      return capsule_owner;
    }
    if (py::isinstance<PyExecuTorchResult>(owner_)) {
      return owner_.cast<std::shared_ptr<PyExecuTorchResult>>()->backing_owner(
          owner_);
    }
    return py::none();
  }

 private:
  static executorch::aten::ScalarType scalar_type_from_dlpack(
      const dlpack::DataType& dtype) {
    if (dtype.lanes != 1) {
      throw py::value_error("DLPack vector dtypes are not supported");
    }
    using executorch::aten::ScalarType;
    if (dtype.code == dlpack::DataTypeCode::Int && dtype.bits == 8) {
      return ScalarType::Char;
    }
    if (dtype.code == dlpack::DataTypeCode::Int && dtype.bits == 16) {
      return ScalarType::Short;
    }
    if (dtype.code == dlpack::DataTypeCode::Int && dtype.bits == 32) {
      return ScalarType::Int;
    }
    if (dtype.code == dlpack::DataTypeCode::Int && dtype.bits == 64) {
      return ScalarType::Long;
    }
    if (dtype.code == dlpack::DataTypeCode::UInt && dtype.bits == 8) {
      return ScalarType::Byte;
    }
    if (dtype.code == dlpack::DataTypeCode::UInt && dtype.bits == 16) {
      return ScalarType::UInt16;
    }
    if (dtype.code == dlpack::DataTypeCode::UInt && dtype.bits == 32) {
      return ScalarType::UInt32;
    }
    if (dtype.code == dlpack::DataTypeCode::UInt && dtype.bits == 64) {
      return ScalarType::UInt64;
    }
    if (dtype.code == dlpack::DataTypeCode::Float && dtype.bits == 16) {
      return ScalarType::Half;
    }
    if (dtype.code == dlpack::DataTypeCode::Float && dtype.bits == 32) {
      return ScalarType::Float;
    }
    if (dtype.code == dlpack::DataTypeCode::Float && dtype.bits == 64) {
      return ScalarType::Double;
    }
    if (dtype.code == dlpack::DataTypeCode::BFloat && dtype.bits == 16) {
      return ScalarType::BFloat16;
    }
    if (dtype.code == dlpack::DataTypeCode::Complex && dtype.bits == 32) {
      return ScalarType::ComplexHalf;
    }
    if (dtype.code == dlpack::DataTypeCode::Complex && dtype.bits == 64) {
      return ScalarType::ComplexFloat;
    }
    if (dtype.code == dlpack::DataTypeCode::Complex && dtype.bits == 128) {
      return ScalarType::ComplexDouble;
    }
    if (dtype.code == dlpack::DataTypeCode::Bool && dtype.bits == 8) {
      return ScalarType::Bool;
    }
    throw py::value_error("Unsupported DLPack dtype");
  }

  void validate_advertised_device() const {
    if (advertised_device_.first != dlpack::CPU &&
        advertised_device_.first != dlpack::CUDA) {
      throw py::value_error("Only CPU and CUDA DLPack tensors are supported");
    }
    if (advertised_device_.second < 0 ||
        advertised_device_.second >
            std::numeric_limits<runtime::etensor::DeviceIndex>::max()) {
      throw py::value_error("DLPack device index is too large for ExecuTorch");
    }
  }

  void initialize(const dlpack::Tensor& tensor) {
    if (static_cast<int>(tensor.device.device_type) !=
            advertised_device_.first ||
        tensor.device.device_id != advertised_device_.second) {
      throw py::value_error(
          "DLPack capsule device disagrees with __dlpack_device__()");
    }
    if (tensor.ndim > 0 && tensor.shape == nullptr) {
      throw py::value_error("DLPack tensor shape is missing");
    }
    if (tensor.ndim < 0 ||
        static_cast<size_t>(tensor.ndim) > runtime::kTensorDimensionLimit) {
      throw py::value_error("DLPack tensor rank is too large for ExecuTorch");
    }
    if (tensor.device.device_type == dlpack::DeviceType::CPU) {
      device_ = executorch::aten::Device(
          executorch::aten::DeviceType::CPU,
          static_cast<runtime::etensor::DeviceIndex>(tensor.device.device_id));
    } else if (tensor.device.device_type == dlpack::DeviceType::CUDA) {
      device_ = executorch::aten::Device(
          executorch::aten::DeviceType::CUDA,
          static_cast<runtime::etensor::DeviceIndex>(tensor.device.device_id));
    } else {
      throw py::value_error("Only CPU and CUDA DLPack tensors are supported");
    }
    scalar_type_ = scalar_type_from_dlpack(tensor.dtype);
    if (tensor.byte_offset > std::numeric_limits<uintptr_t>::max() -
                reinterpret_cast<uintptr_t>(tensor.data) ||
        (tensor.data == nullptr && tensor.byte_offset != 0)) {
      throw py::value_error("Invalid DLPack tensor data pointer");
    }
    data_ = tensor.data == nullptr
        ? nullptr
        : static_cast<uint8_t*>(tensor.data) + tensor.byte_offset;
    sizes_.reserve(tensor.ndim);
    strides_.resize(tensor.ndim);
    for (int32_t i = 0; i < tensor.ndim; ++i) {
      if (tensor.shape[i] < 0 ||
          tensor.shape[i] > std::numeric_limits<int>::max()) {
        throw py::value_error("Invalid DLPack tensor shape");
      }
      sizes_.push_back(static_cast<int>(tensor.shape[i]));
    }
    const bool empty =
        std::find(sizes_.begin(), sizes_.end(), 0) != sizes_.end();
    size_t count = empty ? 0 : 1;
    if (!empty) {
      for (const auto size : sizes_) {
        if (count >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(size)) {
          throw py::value_error("DLPack tensor is too large for ExecuTorch");
        }
        count *= size;
      }
    }
    const auto itemsize = runtime::elementSize(scalar_type_);
    if (count > std::numeric_limits<size_t>::max() / itemsize) {
      throw py::value_error("DLPack tensor is too large for ExecuTorch");
    }
    nbytes_ = count * itemsize;
    if (nbytes_ != 0 && data_ == nullptr) {
      throw py::value_error("DLPack tensor data is not allocated");
    }
    if (tensor.strides == nullptr) {
      int64_t stride = 1;
      for (int32_t i = tensor.ndim; i > 0; --i) {
        if (stride > std::numeric_limits<int>::max()) {
          throw py::value_error(
              "DLPack tensor stride is too large for ExecuTorch");
        }
        strides_[i - 1] = static_cast<int>(stride);
        stride *= std::max(sizes_[i - 1], 1);
      }
    } else {
      for (int32_t i = 0; i < tensor.ndim; ++i) {
        if (tensor.strides[i] < 0 ||
            tensor.strides[i] > std::numeric_limits<int>::max()) {
          throw py::value_error("Invalid DLPack tensor stride");
        }
        strides_[i] = static_cast<int>(tensor.strides[i]);
      }
    }
    dim_order_.resize(sizes_.size());
    std::iota(dim_order_.begin(), dim_order_.end(), 0);
    std::stable_sort(
        dim_order_.begin(), dim_order_.end(), [this](uint8_t a, uint8_t b) {
          return strides_[a] > strides_[b];
        });
    if (empty) {
      return;
    }
    int64_t expected_stride = 1;
    for (size_t i = dim_order_.size(); i > 0; --i) {
      const auto dim = dim_order_[i - 1];
      if (sizes_[dim] > 1 && strides_[dim] != expected_stride) {
        throw py::value_error(
            "ExecuTorch inputs require dense DLPack tensor storage");
      }
      if (expected_stride >
          std::numeric_limits<int>::max() / std::max(sizes_[dim], 1)) {
        throw py::value_error("DLPack tensor is too large for ExecuTorch");
      }
      expected_stride *= std::max(sizes_[dim], 1);
    }
  }

  py::object owner_;
  std::pair<int, int> advertised_device_;
  size_t nbytes_ = 0;
  py::object capsule_;
  dlpack::ManagedTensor* managed_ = nullptr;
  void* data_ = nullptr;
  std::vector<uint8_t> converted_;
  std::vector<int> sizes_;
  std::vector<int> strides_;
  std::vector<uint8_t> dim_order_;
  executorch::aten::ScalarType scalar_type_;
  executorch::aten::Device device_{executorch::aten::DeviceType::CPU};
};

py::object dlpack_output_owner(
    const executorch::aten::Tensor& output,
    const std::vector<std::shared_ptr<DLPackTensorView>>& inputs,
    const std::vector<std::shared_ptr<BufferTensor>>& buffers,
    const std::vector<std::shared_ptr<TorchTensorView>>& torch_inputs,
    const std::vector<std::optional<TensorPtr>>& output_tensors,
    py::object storage_owner) {
  py::list owners;
  std::unordered_set<PyObject*> seen;
  const auto append_owner = [&owners, &seen](py::object owner) {
    // Result feedback should retain allocation roots without nesting the
    // previous call's owner list or duplicating its records.
    std::vector<py::object> pending;
    pending.push_back(std::move(owner));
    while (!pending.empty()) {
      auto current = std::move(pending.back());
      pending.pop_back();
      if (current.is_none() || !seen.insert(current.ptr()).second) {
        continue;
      }
      if (PyList_CheckExact(current.ptr())) {
        for (const auto item : py::reinterpret_borrow<py::list>(current)) {
          pending.emplace_back(py::reinterpret_borrow<py::object>(item));
        }
      } else {
        owners.append(std::move(current));
      }
    }
  };
  append_owner(std::move(storage_owner));
  const auto aliases_output = [&output](const void* data, size_t nbytes) {
    const auto start = reinterpret_cast<uintptr_t>(data);
    const auto address = reinterpret_cast<uintptr_t>(output.const_data_ptr());
    return address != 0 && address >= start && address - start <= nbytes &&
        output.nbytes() <= nbytes - (address - start);
  };
  // Each result pins only its own output allocation. An aliased result must
  // not accumulate allocations belonging to discarded sibling outputs.
  for (const auto& tensor : output_tensors) {
    if (tensor.has_value() &&
        aliases_output((*tensor)->const_data_ptr(), (*tensor)->nbytes())) {
      auto* allocation = new TensorPtr(*tensor);
      append_owner(py::capsule(allocation, [](void* pointer) {
        delete static_cast<TensorPtr*>(pointer);
      }));
    }
  }
  for (const auto& input : inputs) {
    if (!aliases_output(input->data(), input->nbytes())) {
      continue;
    }
    auto backing_owner = input->result_backing_owner();
    if (!backing_owner.is_none()) {
      append_owner(std::move(backing_owner));
    } else {
      auto* retained = new std::shared_ptr<DLPackTensorView>(input);
      append_owner(py::capsule(retained, [](void* pointer) {
        delete static_cast<std::shared_ptr<DLPackTensorView>*>(pointer);
      }));
    }
  }
  for (const auto& buffer : buffers) {
    if (!aliases_output(buffer->data(), buffer->nbytes())) {
      continue;
    }
    auto backing_owner = buffer->result_backing_owner();
    if (!backing_owner.is_none()) {
      append_owner(std::move(backing_owner));
    } else {
      auto* retained = new std::shared_ptr<BufferTensor>(buffer);
      append_owner(py::capsule(retained, [](void* pointer) {
        delete static_cast<std::shared_ptr<BufferTensor>*>(pointer);
      }));
    }
  }
  for (const auto& input : torch_inputs) {
    if (aliases_output(input->data(), input->nbytes())) {
      auto* retained = new std::shared_ptr<TorchTensorView>(input);
      append_owner(py::capsule(retained, [](void* pointer) {
        delete static_cast<std::shared_ptr<TorchTensorView>*>(pointer);
      }));
    }
  }
  return owners;
}

py::object module_aliased_torch_output(
    const executorch::aten::Tensor& tensor,
    const std::vector<std::shared_ptr<TorchTensorView>>& torch_inputs,
    const std::string& method_name) {
  if (torch_inputs.empty() ||
      portable_tensor_output() != PortableTensorOutput::Torch ||
      tensor.nbytes() == 0) {
    return py::none();
  }
  const auto address = reinterpret_cast<uintptr_t>(tensor.const_data_ptr());
  const auto tensor_type =
      py::reinterpret_borrow<py::object>(loaded_torch_module()).attr("Tensor");
  for (const auto& input : torch_inputs) {
    const auto start = reinterpret_cast<uintptr_t>(input->data());
    if (Py_TYPE(input->owner().ptr()) !=
            reinterpret_cast<PyTypeObject*>(tensor_type.ptr()) ||
        tensor.device() != input->device() ||
        tensor.scalar_type() != input->scalar_type() || address < start ||
        address - start > input->nbytes() ||
        tensor.nbytes() > input->nbytes() - (address - start)) {
      continue;
    }
    const auto itemsize = runtime::elementSize(tensor.scalar_type());
    if ((address - start) % itemsize != 0) {
      continue;
    }
    auto source = input->owner().attr("detach")();
    if (source.attr("data_ptr")().cast<uintptr_t>() != start ||
        scalar_type_from_torch_dtype(
            py::str(source.attr("dtype")).cast<std::string>()) !=
            tensor.scalar_type()) {
      throw py::value_error(
          "Input " + std::to_string(input->index()) + " of method '" +
          method_name +
          "': its storage changed before the output was read; pass it again");
    }
    const auto offset = source.attr("storage_offset")().cast<int64_t>() +
        static_cast<int64_t>((address - start) / itemsize);
    return source.attr("as_strided")(
        std::vector<int64_t>(tensor.sizes().begin(), tensor.sizes().end()),
        std::vector<int64_t>(tensor.strides().begin(), tensor.strides().end()),
        offset);
  }
  return py::none();
}
#endif

// The value for an input the method declares as a number. Python numbers and
// NumPy scalars or 0-d arrays must match the declared kind, except that an
// integer is accepted for a float input. NumPy numbers also export a
// buffer, so without this they would be read as tensors. Returns an empty
// optional for inputs the method does not declare as numbers, and raises for
// anything else passed where a number is expected.
std::optional<EValue> scalar_input(
    const MethodMeta& method_meta,
    size_t index,
    const py::handle& value) {
  const auto tag = method_meta.input_tag(index);
  if (!tag.ok() ||
      (tag.get() != Tag::Double && tag.get() != Tag::Int &&
       tag.get() != Tag::Bool)) {
    return std::nullopt;
  }
  py::object number;
  PyObject* numpy = PyDict_GetItemString(PyImport_GetModuleDict(), "numpy");
  if (py::isinstance<py::bool_>(value) || py::isinstance<py::int_>(value) ||
      py::isinstance<py::float_>(value)) {
    number = py::reinterpret_borrow<py::object>(value);
  } else if (numpy != nullptr) {
    const auto module = py::reinterpret_borrow<py::object>(numpy);
    if (py::isinstance(value, module.attr("generic")) ||
        (py::isinstance(value, module.attr("ndarray")) &&
         value.attr("ndim").cast<int>() == 0)) {
      number = value.attr("item")();
    }
  }
  const bool is_bool = number && py::isinstance<py::bool_>(number);
  // Python's bool is a subclass of int.
  const bool is_int = number && !is_bool && py::isinstance<py::int_>(number);
  const bool is_float = number && py::isinstance<py::float_>(number);
  if (tag.get() == Tag::Double && (is_int || is_float)) {
    return EValue(number.cast<double>());
  }
  if (tag.get() == Tag::Int && is_int) {
    return EValue(number.cast<int64_t>());
  }
  if (tag.get() == Tag::Bool && is_bool) {
    return EValue(number.cast<bool>());
  }
  throw std::runtime_error(
      std::string("the method expects ") + runtime::tag_to_string(tag.get()) +
      ", but got " + std::string(py::str(value.get_type())));
}

bool prefer_dlpack_input(const py::handle& value) {
  if (!py::hasattr(value, "__dlpack__")) {
    return false;
  }
  if (!PyObject_CheckBuffer(value.ptr())) {
    return true;
  }
  // A result of these bindings always takes the DLPack path, which keeps its
  // real dtype, shape, and strides. A bfloat16 result exports flat bytes
  // instead, which the buffer path would read in the method's layout.
  if (py::isinstance<PyExecuTorchResult>(value)) {
    return true;
  }
  // NumPy and other host buffers keep the buffer path, including its
  // read-only copy behavior. CUDA results expose only their DLPack storage.
  if (py::hasattr(value, "__dlpack_device__")) {
    const auto device =
        value.attr("__dlpack_device__")().cast<std::pair<int, int>>();
    return device.first == dlpack::CUDA;
  }
  return false;
}

py::sequence normalize_inputs(const py::object& inputs) {
  if (PyObject_CheckBuffer(inputs.ptr()) || is_torch_tensor(inputs) ||
      py::hasattr(inputs, "__dlpack__")) {
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
std::shared_ptr<DeviceMemoryBuffer> allocate_device_output(
    size_t index,
    size_t nbytes,
    executorch::aten::Device device) {
  auto buffer =
      DeviceMemoryBuffer::create(nbytes, device.type(), device.index());
  THROW_IF_ERROR(buffer.error(), "Failed to allocate device output %zu", index);
  return std::make_shared<DeviceMemoryBuffer>(std::move(buffer.get()));
}

// The Python boundary is synchronous, and each wait is for the whole device,
// so one wait per device covers every tensor of a run. Before the run, another
// library may still be writing a CUDA input on its own streams: a torch tensor,
// a DLPack tensor, or a result of these bindings, which can share memory a
// caller gave to that library. It may also still be reading the memory of an
// output from an earlier run, such as a dropped result's buffer or a
// memory-planned output, which this run writes again. After the run, its
// outputs, and any input it updated in place, are ready on every stream.
void synchronize_tensor_devices(const Method& method) {
  std::vector<int> devices;
  const auto add = [&devices](const EValue& value) {
    if (!value.isTensor() || value.toTensor().nbytes() == 0) {
      return;
    }
    const auto device = value.toTensor().device();
    if (!device.is_cpu() &&
        std::find(devices.begin(), devices.end(), device.index()) ==
            devices.end()) {
      devices.push_back(device.index());
    }
  };
  for (size_t i = 0; i < method.inputs_size(); ++i) {
    add(method.get_input(i));
  }
  for (size_t i = 0; i < method.outputs_size(); ++i) {
    add(method.get_output(i));
  }
  for (const auto index : devices) {
    synchronize_device(index);
  }
}

py::object portable_tensor_result(
    const executorch::aten::Tensor& tensor,
    bool clone = true,
    py::object owner = py::none()) {
  if (portable_tensor_output() == PortableTensorOutput::ExecuTorch) {
    return py::cast(std::make_shared<PyExecuTorchResult>(
        tensor, clone, clone ? py::none() : std::move(owner)));
  }

  PyObject* loaded_torch = loaded_torch_module();
  if (loaded_torch == nullptr) {
    throw std::runtime_error(
        "PyTorch tensor output was requested, but torch is not imported. "
        "Import torch before executing the portable bindings or set "
        "EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT=executorch before importing "
        "them to receive ExecuTorchResult outputs.");
  }

  const auto torch_module = py::reinterpret_borrow<py::object>(loaded_torch);
  const auto dtype =
      torch_module.attr(python_dtype(tensor.scalar_type()).torch_name);
  if (!tensor.device().is_cpu()) {
    const auto shape = py::cast(
        std::vector<int64_t>(tensor.sizes().begin(), tensor.sizes().end()));
    const auto strides = py::cast(
        std::vector<int64_t>(tensor.strides().begin(), tensor.strides().end()));
    if (tensor.nbytes() == 0) {
      return torch_module.attr("empty_strided")(
          shape,
          strides,
          py::arg("dtype") = dtype,
          py::arg("device") =
              "cuda:" + std::to_string(tensor.device().index()));
    }
    if (tensor.const_data_ptr() == nullptr) {
      throw py::value_error("CUDA output tensor data is not allocated");
    }
    // A byte capsule supports Torch dtypes that older DLPack dtype tables
    // cannot express. Torch reinterprets the same device storage afterward.
    auto output = torch_module
                      .attr("from_dlpack")(
                          TorchDLPackBuffer::create(tensor, std::move(owner)))
                      .attr("view")(dtype)
                      .attr("as_strided")(shape, strides);
    if (clone) {
      // Torch copies on its current stream; the copy must finish before the
      // method's next execution writes the memory it reads.
      output = output.attr("clone")();
      synchronize_torch_cuda_device(tensor.device().index());
    }
    return output;
  }
  auto result = std::make_shared<PyExecuTorchResult>(
      tensor, clone, clone ? py::none() : std::move(owner));
  if (result->nbytes() == 0) {
    return torch_module.attr("empty_strided")(
        result->shape(),
        result->element_strides(),
        py::arg("dtype") = dtype,
        py::arg("device") = "cpu");
  }
  const auto shape = result->shape();
  const auto strides = result->element_strides();
  py::object flat_buffer = py::cast(
      std::make_shared<PyExecuTorchResultFlatBuffer>(std::move(result)));
  return torch_module.attr("frombuffer")(flat_buffer, py::arg("dtype") = dtype)
      .attr("as_strided")(shape, strides);
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

// BufferDataLoader borrows the Python bytes, and results and method metadata
// can share the module after its Python wrapper is gone. Owning the bytes
// through the shared module keeps them alive for every holder.
inline std::shared_ptr<Module> share_module_with_buffers(
    std::unique_ptr<Module> module,
    py::bytes program,
    std::optional<py::bytes> data_map) {
  struct Owner final {
    py::bytes program;
    std::optional<py::bytes> data_map;
    std::unique_ptr<Module> module;
  };
  auto owner = std::make_shared<Owner>(
      Owner{std::move(program), std::move(data_map), std::move(module)});
  return std::shared_ptr<Module>(owner, owner->module.get());
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
    bool clone_outputs = true,
    py::object owner = py::none(),
    const std::vector<py::object>& output_owners = {},
    const std::vector<py::object>& handed_over = {}) {
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
      const bool returns_without_copy =
          i < handed_over.size() && handed_over[i];
      list[i] = returns_without_copy
          ? portable_tensor_result(
                v.toTensor(), /*clone=*/false, handed_over[i])
          : portable_tensor_result(
                v.toTensor(),
                clone_outputs,
                output_owners.empty() ? owner : output_owners[i]);
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

  const py::bytes& get_bundled_program_bytes() const {
    return bundled_program_ptr_;
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
#ifndef USE_ATEN_LIB
    {
      const auto method = this->method(method_name);
      THROW_IF_ERROR(
          method.error(),
          "Method execution failed with status 0x%" PRIx32,
          static_cast<uint32_t>(method.error()));
      synchronize_tensor_devices(*method.get());
    }
#endif

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
  // BufferDataLoader borrows this storage. Declare its owner first so it is
  // destroyed after the program and loaders that read from it.
  std::optional<py::bytes> program_buffer_;
  std::unique_ptr<DataLoader> loader_;
  std::unique_ptr<Program> program_;
  // Owned here rather than by PyProgram, beside the loader it reads
  // through, because a Method borrows the map and outlives the call that
  // loaded it. Both stay alive as long as any method does.
  std::unique_ptr<DataLoader> data_map_loader_;
  std::unique_ptr<FlatTensorDataMap> data_map_;
  // Load options for every method of this program. The map's spans point into
  // the storage.
  std::vector<std::vector<BackendOption>> backend_option_storage_;
  LoadBackendOptionsMap backend_options_;

  explicit ProgramState(
      std::unique_ptr<DataLoader> loader,
      std::unique_ptr<Program> program,
      std::unique_ptr<DataLoader> data_map_loader = nullptr,
      std::unique_ptr<FlatTensorDataMap> data_map = nullptr,
      std::optional<py::bytes> program_buffer = std::nullopt)
      : program_buffer_(std::move(program_buffer)),
        loader_(std::move(loader)),
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
      : PyModule(
            buffer,
            buffer.cast<std::string_view>().data(),
            py::len(buffer),
            std::move(data_map_buffer),
            enable_etdump,
            debug_buffer_size,
            program_verification) {}

  // The program is read from `program_owner`, which the module keeps alive.
  explicit PyModule(
      const py::bytes& program_owner,
      const void* ptr,
      size_t ptr_len,
      std::optional<const py::bytes> data_map_buffer,
      bool enable_etdump,
      size_t debug_buffer_size = 0,
      Program::Verification program_verification =
          Program::Verification::InternalConsistency)
      : debug_buffer_size_(debug_buffer_size),
        module_(share_module_with_buffers(
            load_module_from_buffer(
                ptr,
                ptr_len,
                data_map_buffer.has_value()
                    ? std::optional<const void*>(data_map_buffer.value()
                                                     .cast<std::string_view>()
                                                     .data())
                    : std::nullopt,
                data_map_buffer.has_value()
                    ? std::optional<size_t>(py::len(data_map_buffer.value()))
                    : std::nullopt,
                setup_event_tracer(enable_etdump, debug_buffer_size),
                program_verification),
            program_owner,
            data_map_buffer)) {}

  // The program is read from `program_owner`, which the module keeps alive.
  explicit PyModule(
      const py::bytes& program_owner,
      const void* ptr,
      size_t ptr_len,
      const std::string& data_path,
      bool enable_etdump,
      size_t debug_buffer_size = 0,
      Program::Verification program_verification =
          Program::Verification::InternalConsistency)
      : debug_buffer_size_(debug_buffer_size),
        module_(share_module_with_buffers(
            load_module_from_buffer_with_data_file(
                ptr,
                ptr_len,
                data_path,
                setup_event_tracer(enable_etdump, debug_buffer_size),
                program_verification),
            program_owner,
            std::nullopt)) {}

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
  PyModule(PyModule&&) = delete;
  PyModule& operator=(PyModule&&) = delete;

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
    return std::make_unique<PyModule>(
        m.get_bundled_program_bytes(),
        m.get_program_ptr(),
        m.get_program_len(),
        data_map_buffer,
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
        m.get_bundled_program_bytes(),
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

  struct InputOwner final {
    std::shared_ptr<BufferTensor> buffer;
#ifndef USE_ATEN_LIB
    std::shared_ptr<TorchTensorView> torch;
    std::shared_ptr<DLPackTensorView> dlpack;
#else
    py::object torch;
#endif
    std::optional<BorrowedInput> validation;
  };

  py::object native_output_owner() {
    if (!native_output_owner_) {
      auto* module = new std::shared_ptr<Module>(module_);
      native_output_owner_ = py::capsule(module, [](void* pointer) {
        delete static_cast<std::shared_ptr<Module>*>(pointer);
      });
    }
    return native_output_owner_;
  }

  py::list run_method(
      const std::string& method_name,
      const py::object& python_inputs,
      bool clone_outputs = true) {
    ObjectLock lock(mutex_);
    const auto inputs = normalize_inputs(python_inputs);
    const auto inputs_size = py::len(inputs);
    const auto method_meta_result = module_->method_meta(method_name);
    THROW_IF_ERROR(
        method_meta_result.error(),
        "Failed to get metadata for method %s",
        method_name.c_str());
    const auto method_meta = method_meta_result.get();
    // Checked before anything is installed: an extra input would otherwise
    // install every real one and then fail, blocking later partial calls.
    if (inputs_size > method_meta.num_inputs()) {
      throw std::runtime_error(
          "Method '" + method_name + "' takes " +
          std::to_string(method_meta.num_inputs()) + " inputs, but got " +
          std::to_string(inputs_size));
    }
    if (inputs_size < method_meta.num_inputs() &&
        failed_methods_.count(method_name) != 0) {
      throw std::runtime_error(
          "The last call to method '" + method_name +
          "' failed while setting its inputs, so the inputs it would reuse "
          "may be a mix of two calls. Pass every input on the next call.");
    }
#ifndef USE_ATEN_LIB
    const auto loaded_method = module_->method(method_name);
    THROW_IF_ERROR(
        loaded_method.error(),
        "Failed to execute method %s, error: 0x%" PRIx32,
        method_name.c_str(),
        static_cast<uint32_t>(loaded_method.error()));
    const Method& method = *loaded_method.get();
#endif
    std::vector<InputOwner> new_input_owners(inputs_size);
    std::vector<EValue> cpp_inputs;
    cpp_inputs.reserve(inputs_size);
    std::vector<std::shared_ptr<BufferTensor>> buffer_inputs;
    buffer_inputs.reserve(inputs_size);
#ifndef USE_ATEN_LIB
    std::vector<std::shared_ptr<TorchTensorView>> torch_inputs;
    torch_inputs.reserve(inputs_size);
    std::vector<std::shared_ptr<DLPackTensorView>> dlpack_inputs;
    dlpack_inputs.reserve(inputs_size);
    std::vector<TensorPtr> dlpack_tensor_ptrs;
    dlpack_tensor_ptrs.reserve(inputs_size);
#endif

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
      if (auto scalar = read_input(method_meta, i, [&] {
            return scalar_input(method_meta, i, python_input);
          })) {
        cpp_inputs.push_back(*scalar);
      } else if (read_input(method_meta, i, [&] {
                   return is_torch_tensor(python_input);
                 })) {
#ifdef USE_ATEN_LIB
        auto at_tensor = python_input.cast<at::Tensor>();
        new_input_owners[i].torch =
            py::reinterpret_borrow<py::object>(python_input);
        std::vector<int> tensor_sizes(
            at_tensor.sizes().begin(), at_tensor.sizes().end());
        validate_tensor_input(
            method_meta, i, at_tensor.scalar_type(), tensor_sizes);
        (void)mutable_tensor_data_ptr_no_cow(at_tensor);
        cpp_inputs.emplace_back(at_tensor);
        if (!method_meta.input_tensor_meta(i)->is_memory_planned()) {
          new_input_owners[i].validation.emplace(i, python_input);
        }
#else
        torch_inputs.push_back(read_input(method_meta, i, [&] {
          return std::make_shared<TorchTensorView>(python_input, i);
        }));
        const auto& tensor = torch_inputs.back();
        new_input_owners[i].torch = tensor;
        validate_tensor_input(
            method_meta, i, tensor->scalar_type(), tensor->sizes());
        validate_input_device(method, i, tensor->device(), tensor->nbytes());
        read_input(method_meta, i, [&] {
          tensor->use_layout(method_meta.input_tensor_meta(i)->dim_order());
        });
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
            torch::executor::TensorShapeDynamism::STATIC,
            tensor->device().type(),
            tensor->device().index());
        cpp_inputs.emplace_back(torch::executor::Tensor(&input_tensors.back()));
        if (!method_meta.input_tensor_meta(i)->is_memory_planned()) {
          new_input_owners[i].validation.emplace(i, tensor->owner());
        }
#endif
      } else if (py::isinstance<py::none>(python_input)) {
        cpp_inputs.push_back(EValue());
      } else if (py::isinstance<py::bool_>(python_input)) {
        cpp_inputs.push_back(EValue(py::cast<bool>(python_input)));
      } else if (py::isinstance<py::int_>(python_input)) {
        cpp_inputs.push_back(EValue(py::cast<int64_t>(python_input)));
      } else if (py::isinstance<py::float_>(python_input)) {
        cpp_inputs.push_back(EValue(py::cast<double>(python_input)));
      } else if (read_input(method_meta, i, [&] {
                   return prefer_dlpack_input(python_input);
                 })) {
#ifdef USE_ATEN_LIB
        auto python_tensor = py::module_::import("torch.utils.dlpack")
                                 .attr("from_dlpack")(python_input);
        auto at_tensor = python_tensor.cast<at::Tensor>();
        new_input_owners[i].torch = python_tensor;
        validate_tensor_input(
            method_meta,
            i,
            at_tensor.scalar_type(),
            std::vector<int>(
                at_tensor.sizes().begin(), at_tensor.sizes().end()));
        cpp_inputs.emplace_back(at_tensor);
#else
        dlpack_inputs.push_back(read_input(method_meta, i, [&] {
          return std::make_shared<DLPackTensorView>(python_input);
        }));
        const auto& view = dlpack_inputs.back();
        new_input_owners[i].dlpack = view;
        validate_tensor_input(
            method_meta, i, view->scalar_type(), view->sizes());
        validate_input_device(method, i, view->device(), view->nbytes());
        read_input(method_meta, i, [&] {
          view->use_layout(method_meta.input_tensor_meta(i)->dim_order());
        });
        auto tensor = for_blob(view->data(), view->sizes(), view->scalar_type())
                          .strides(view->strides())
                          .dim_order(view->dim_order())
                          .dynamism(aten::TensorShapeDynamism::STATIC)
                          .device(view->device())
                          .make_tensor_ptr();
        dlpack_tensor_ptrs.push_back(std::move(tensor));
        cpp_inputs.emplace_back(dlpack_tensor_ptrs.back());
#endif
      } else if (PyObject_CheckBuffer(python_input.ptr())) {
        buffer_inputs.push_back(read_input(method_meta, i, [&] {
          return std::make_shared<BufferTensor>(python_input, method_meta, i);
        }));
        const auto& buffer = buffer_inputs.back();
        new_input_owners[i].buffer = buffer;
#ifndef USE_ATEN_LIB
        validate_input_device(
            method,
            i,
            executorch::aten::Device(executorch::aten::DeviceType::CPU),
            buffer->nbytes());
#endif
        validate_buffer_input(method_meta, i, *buffer);
        if (!method_meta.input_tensor_meta(i)->is_memory_planned() &&
            buffer->borrows_data()) {
          new_input_owners[i].validation.emplace(i, buffer);
        }
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
            "Input " + std::to_string(i) + " of method '" + method_meta.name() +
            "': unsupported python type " + type_str +
            ". Ensure that inputs are passed as a flat list of tensors.");
      }
    }

    // Module keeps installed inputs for calls with an omitted or partial
    // input list, so their owners stay alive with it. A call that fails while
    // installing keeps nothing: it may have installed a prefix, but later
    // calls must pass every input again, so nothing reads that prefix.
    auto& retained = retained_inputs_[method_name];
    for (size_t i = inputs_size; i < retained.size(); ++i) {
      if (retained[i].validation) {
        retained[i].validation->validate(method_name.c_str());
      }
    }
    // A load error is the real cause, so it wins over the checks below.
    const auto load_status = module_->load_method(method_name);
    THROW_IF_ERROR(
        load_status,
        "Failed to execute method %s, error: 0x%" PRIx32,
        method_name.c_str(),
        static_cast<uint32_t>(load_status));
    // allocate_output_tensors() reads inputs through Method::get_input(),
    // which marks every input as set and hides the runtime's own check for
    // inputs that were never passed.
    const auto passed_inputs =
        std::max(retained.size(), static_cast<size_t>(inputs_size));
    if (passed_inputs < method_meta.num_inputs()) {
      throw std::runtime_error(
          "Method '" + method_name + "' takes " +
          std::to_string(method_meta.num_inputs()) + " inputs, but only " +
          std::to_string(passed_inputs) +
          " were ever passed. Pass every input.");
    }
    allocate_output_tensors(method_name);
    // Install the inputs first: only a failure there can leave a mix of two
    // calls installed, since the runtime does not roll back a prefix.
    for (size_t i = 0; i < inputs_size; ++i) {
      const auto status = module_->set_input(method_name, cpp_inputs[i], i);
      if (status != Error::Ok) {
        failed_methods_.insert(method_name);
        THROW_IF_ERROR(
            status,
            "Failed to set input %zu of method %s, error: 0x%" PRIx32,
            i,
            method_name.c_str(),
            static_cast<uint32_t>(status));
      }
    }
    failed_methods_.erase(method_name);
    retained.resize(passed_inputs);
    for (size_t i = 0; i < inputs_size; ++i) {
      retained[i] = std::move(new_input_owners[i]);
    }
#ifndef USE_ATEN_LIB
    // The run reads inputs kept from earlier calls too, so the wait and the
    // output owners below cover every installed input, not only this call's.
    buffer_inputs.clear();
    torch_inputs.clear();
    dlpack_inputs.clear();
    for (const auto& owner : retained) {
      if (owner.buffer) {
        buffer_inputs.push_back(owner.buffer);
      }
      if (owner.torch) {
        torch_inputs.push_back(owner.torch);
      }
      if (owner.dlpack) {
        dlpack_inputs.push_back(owner.dlpack);
      }
    }
    synchronize_tensor_devices(method);
#endif
    auto outputs = module_->execute(method_name);
    THROW_IF_ERROR(
        outputs.error(),
        "Failed to execute method %s, error: 0x%" PRIx32,
        method_name.c_str(),
        static_cast<uint32_t>(outputs.error()));
#ifndef USE_ATEN_LIB
    synchronize_tensor_devices(method);
#endif
    // DLPack producers may supply storage returned directly by the graph.
    // Keep each output's allocation alive after local adapters go out of scope.
    py::object output_owner = native_output_owner();
    std::vector<py::object> output_owners;
#ifndef USE_ATEN_LIB
    output_owners.reserve(outputs->size());
    for (const auto& output : outputs.get()) {
      output_owners.push_back(
          output.isTensor() ? dlpack_output_owner(
                                  output.toTensor(),
                                  dlpack_inputs,
                                  buffer_inputs,
                                  torch_inputs,
                                  output_tensors_,
                                  output_owner)
                            : py::none());
    }
#endif
    auto result = get_outputs_as_py_list(
        outputs.get(),
        clone_outputs,
        std::move(output_owner),
        output_owners,
        clone_outputs ? handed_over_device_outputs(outputs.get())
                      : std::vector<py::object>());
#ifndef USE_ATEN_LIB
    if (!clone_outputs) {
      for (size_t i = 0; i < outputs->size(); ++i) {
        const auto& output = (*outputs)[i];
        if (output.isTensor()) {
          auto alias = module_aliased_torch_output(
              output.toTensor(), torch_inputs, method_name);
          if (!alias.is_none()) {
            result[i] = std::move(alias);
          }
        }
      }
    }
#endif
    return result;
  }

  py::list forward(const py::object& inputs, bool clone_outputs = true) {
    return run_method("forward", inputs, clone_outputs);
  }

  bool has_etdump() {
    ObjectLock lock(mutex_);
    ETDumpGen* etdump = dynamic_cast<ETDumpGen*>(module_->event_tracer());
    return etdump != nullptr;
  }

  void write_etdump_result_to_file(
      const std::string& path,
      const py::object& debug_buffer_path) {
    ObjectLock lock(mutex_);
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
    ObjectLock lock(mutex_);
    return run_method(method_name, py::list(), clone_outputs);
  }

  std::unique_ptr<PyMethodMeta> method_meta(const std::string method_name) {
    ObjectLock lock(mutex_);
    auto method_data = module_->method_meta(method_name);
    THROW_IF_ERROR(
        method_data.error(),
        "failed to retrieve method_meta for method %s, error 0x%" PRIx32,
        method_name.c_str(),
        static_cast<uint32_t>(method_data.error()));
    return std::make_unique<PyMethodMeta>(module_, method_data.get());
  }

  std::vector<std::string> method_names() {
    ObjectLock lock(mutex_);
    auto result = module_->method_names();
    THROW_IF_ERROR(
        result.error(),
        "Failed to get method names, error: 0x%" PRIx32,
        static_cast<uint32_t>(result.error()));
    const auto& method_set = result.get();
    return std::vector<std::string>(method_set.begin(), method_set.end());
  }

 private:
  std::recursive_mutex mutex_;
  // Hold onto the debug_buffer_ for the event_tracer.
  std::unique_ptr<uint8_t[]> debug_buffer_;
  size_t debug_buffer_size_;

  std::shared_ptr<Module> module_;
  std::unordered_map<std::string, std::vector<InputOwner>> retained_inputs_;
  // Methods whose last call failed while installing its inputs.
  std::unordered_set<std::string> failed_methods_;
  py::object native_output_owner_;
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

  // allocate_output_tensors() gives each call new device memory, so a result
  // can keep it instead of a copy.
  std::vector<py::object> handed_over_device_outputs(
      const std::vector<EValue>& outputs) {
    std::vector<py::object> owned(outputs.size());
#ifndef USE_ATEN_LIB
    for (size_t i = 0; i < outputs.size() && i < output_tensors_.size(); ++i) {
      if (!outputs[i].isTensor() || !output_tensors_[i].has_value()) {
        continue;
      }
      const auto& tensor = outputs[i].toTensor();
      const auto& allocation = *output_tensors_[i];
      if (!tensor.device().is_cpu() && tensor.nbytes() != 0 &&
          allocation->const_data_ptr() == tensor.const_data_ptr() &&
          allocation->nbytes() == tensor.nbytes()) {
        auto* retained = new TensorPtr(allocation);
        owned[i] = py::capsule(retained, [](void* pointer) {
          delete static_cast<TensorPtr*>(pointer);
        });
      }
    }
#endif
    return owned;
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
      const auto method = module_->method(method_name);
      THROW_IF_ERROR(
          method.error(),
          "Failed to load method %s, error: 0x%" PRIx32,
          method_name.c_str(),
          static_cast<uint32_t>(method.error()));
      const auto& output = method.get()->get_output(i);
      bool is_input = false;
      for (size_t input = 0; input < method.get()->inputs_size(); ++input) {
        is_input |= &output == &method.get()->get_input(input);
      }
      if (is_input) {
        // The shared value receives its storage from set_inputs().
        output_tensors_.emplace_back(std::nullopt);
        continue;
      }
#ifndef USE_ATEN_LIB
      const auto device = output.toTensor().device();
      if (!device.is_cpu()) {
        std::shared_ptr<DeviceMemoryBuffer> allocation;
        const auto nbytes = output_tensor_meta->nbytes();
        if (nbytes != 0) {
          allocation = allocate_device_output(i, nbytes, device);
        }
        std::vector<executorch::aten::SizesType> sizes(
            output_tensor_meta->sizes().begin(),
            output_tensor_meta->sizes().end());
        std::vector<executorch::aten::DimOrderType> order(
            output_tensor_meta->dim_order().begin(),
            output_tensor_meta->dim_order().end());
        auto tensor = for_blob(
                          allocation ? allocation->data() : nullptr,
                          std::move(sizes),
                          output_tensor_meta->scalar_type())
                          .dim_order(std::move(order))
                          .device(device)
                          .deleter([allocation](void*) {})
                          .make_tensor_ptr();
        output_tensors_.emplace_back(std::move(tensor));
        continue;
      }
#endif
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

std::vector<BackendOption> to_backend_options(const py::handle& options) {
  if (!py::isinstance<py::dict>(options)) {
    throw py::type_error("Backend options must be a dict");
  }
  std::vector<BackendOption> result;
  for (const auto& [key, value] : py::reinterpret_borrow<py::dict>(options)) {
    BackendOption option{};
    const auto name = py::cast<std::string>(key);
    if (name.size() >= kMaxOptionKeyLength) {
      throw py::value_error("Backend option key is too long: " + name);
    }
    std::strncpy(option.key, name.c_str(), kMaxOptionKeyLength - 1);
    // A Python bool is also an int, so it is checked first.
    if (py::isinstance<py::bool_>(value)) {
      option.value = py::cast<bool>(value);
    } else if (py::isinstance<py::int_>(value)) {
      const auto number = py::cast<int64_t>(value);
      if (number < std::numeric_limits<int>::min() ||
          number > std::numeric_limits<int>::max()) {
        throw py::value_error(
            "Backend option " + name + " does not fit an int");
      }
      option.value = static_cast<int>(number);
    } else if (py::isinstance<py::str>(value)) {
      const auto text = py::cast<std::string>(value);
      if (text.size() >= kMaxOptionValueLength) {
        throw py::value_error("Backend option " + name + " is too long");
      }
      std::array<char, kMaxOptionValueLength> chars{};
      std::strncpy(chars.data(), text.c_str(), kMaxOptionValueLength - 1);
      option.value = chars;
    } else {
      throw py::type_error(
          "Backend option " + name + " must be a bool, int or str");
    }
    result.push_back(option);
  }
  return result;
}

inline std::shared_ptr<ProgramState> load_program(
    std::unique_ptr<DataLoader> loader,
    Program::Verification program_verification,
    std::optional<const std::string> data_path = std::nullopt,
    std::optional<py::bytes> program_buffer = std::nullopt,
    const py::dict& backend_options = py::dict()) {
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
  auto state = std::make_shared<ProgramState>(
      std::move(loader),
      std::make_unique<Program>(std::move(res.get())),
      std::move(data_map_loader),
      std::move(data_map),
      std::move(program_buffer));
  // Reserved up front so adding an entry never moves the ones the map spans.
  state->backend_option_storage_.reserve(backend_options.size());
  for (const auto& [backend, options] : backend_options) {
    const auto backend_name = py::cast<std::string>(backend);
    auto& owned = state->backend_option_storage_.emplace_back(
        to_backend_options(options));
    const Error error = state->backend_options_.set_options(
        backend_name.c_str(), Span<BackendOption>(owned.data(), owned.size()));
    THROW_IF_ERROR(
        error,
        "Failed to set load options for backend %s, error: 0x%" PRIx32,
        backend_name.c_str(),
        static_cast<uint32_t>(error));
  }
  return state;
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

  bool owns_data(const void* data) const {
    const auto address = reinterpret_cast<uintptr_t>(data);
    for (const auto& buffer : non_const_spans_) {
      const auto start = reinterpret_cast<uintptr_t>(buffer.data());
      if (address != 0 && address >= start && address - start < buffer.size()) {
        return true;
      }
    }
    return false;
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
      : storage_(std::make_shared<Storage>(
            std::move(memory),
            std::move(state),
            std::move(method))) {
    inputs_set_ = storage_->method->inputs_size() == 0;
  }

  void set_inputs(const py::object& python_inputs) {
    ObjectLock lock(mutex_);
    const auto inputs = normalize_inputs(python_inputs);
    const auto inputs_size = py::len(inputs);
    if (inputs_size != storage_->method->inputs_size()) {
      throw std::runtime_error(
          "Method '" + std::string(storage_->method->method_meta().name()) +
          "' takes " + std::to_string(storage_->method->inputs_size()) +
          " inputs, but got " + std::to_string(inputs_size));
    }
    std::vector<EValue> cpp_inputs;
    cpp_inputs.reserve(inputs_size);
    std::vector<std::shared_ptr<BufferTensor>> buffer_inputs;
    buffer_inputs.reserve(inputs_size);
    std::vector<TensorPtr> buffer_tensor_ptrs;
    buffer_tensor_ptrs.reserve(inputs_size);
    std::vector<BorrowedInput> borrowed_inputs;
    borrowed_inputs.reserve(inputs_size);
#ifndef USE_ATEN_LIB
    std::vector<std::shared_ptr<TorchTensorView>> torch_inputs;
    torch_inputs.reserve(inputs_size);
    std::vector<std::shared_ptr<DLPackTensorView>> dlpack_inputs;
    dlpack_inputs.reserve(inputs_size);
    std::vector<TensorPtr> dlpack_tensor_ptrs;
    dlpack_tensor_ptrs.reserve(inputs_size);
#endif

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
      if (auto scalar = read_input(storage_->method->method_meta(), i, [&] {
            return scalar_input(
                storage_->method->method_meta(), i, python_input);
          })) {
        cpp_inputs.push_back(*scalar);
      } else if (read_input(storage_->method->method_meta(), i, [&] {
                   return is_torch_tensor(python_input);
                 })) {
#ifdef USE_ATEN_LIB
        auto at_tensor = python_input.cast<at::Tensor>();
        std::vector<int> tensor_sizes(
            at_tensor.sizes().begin(), at_tensor.sizes().end());
        validate_tensor_input(
            storage_->method->method_meta(),
            i,
            at_tensor.scalar_type(),
            tensor_sizes);
        const auto input_meta =
            storage_->method->method_meta().input_tensor_meta(i).get();
        if (!input_meta.is_memory_planned()) {
          borrowed_inputs.emplace_back(i, python_input);
        }
        (void)mutable_tensor_data_ptr_no_cow(at_tensor);
        cpp_inputs.emplace_back(at_tensor);
#else
        torch_inputs.push_back(
            read_input(storage_->method->method_meta(), i, [&] {
              return std::make_shared<TorchTensorView>(python_input, i);
            }));
        const auto& view = torch_inputs.back();
        validate_tensor_input(
            storage_->method->method_meta(),
            i,
            view->scalar_type(),
            view->sizes());
        validate_input_device(
            *storage_->method, i, view->device(), view->nbytes());
        read_input(storage_->method->method_meta(), i, [&] {
          view->use_layout(storage_->method->method_meta()
                               .input_tensor_meta(i)
                               ->dim_order());
        });
        const auto input_meta =
            storage_->method->method_meta().input_tensor_meta(i).get();
        if (!input_meta.is_memory_planned()) {
          // Watch the tensor the method reads: after a layout conversion that
          // is the copy, which an output aliasing the input can expose.
          borrowed_inputs.emplace_back(i, view->owner());
        }
        auto tensor = for_blob(view->data(), view->sizes(), view->scalar_type())
                          .strides(view->strides())
                          .dim_order(view->dim_order())
                          .dynamism(aten::TensorShapeDynamism::STATIC)
                          .device(view->device())
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
      } else if (read_input(storage_->method->method_meta(), i, [&] {
                   return prefer_dlpack_input(python_input);
                 })) {
#ifdef USE_ATEN_LIB
        auto python_tensor = py::module_::import("torch.utils.dlpack")
                                 .attr("from_dlpack")(python_input);
        auto at_tensor = python_tensor.cast<at::Tensor>();
        validate_tensor_input(
            storage_->method->method_meta(),
            i,
            at_tensor.scalar_type(),
            std::vector<int>(
                at_tensor.sizes().begin(), at_tensor.sizes().end()));
        if (!storage_->method->method_meta()
                 .input_tensor_meta(i)
                 ->is_memory_planned()) {
          borrowed_inputs.emplace_back(i, python_tensor);
        }
        cpp_inputs.emplace_back(at_tensor);
#else
        dlpack_inputs.push_back(
            read_input(storage_->method->method_meta(), i, [&] {
              return std::make_shared<DLPackTensorView>(python_input);
            }));
        const auto& view = dlpack_inputs.back();
        validate_tensor_input(
            storage_->method->method_meta(),
            i,
            view->scalar_type(),
            view->sizes());
        validate_input_device(
            *storage_->method, i, view->device(), view->nbytes());
        read_input(storage_->method->method_meta(), i, [&] {
          view->use_layout(storage_->method->method_meta()
                               .input_tensor_meta(i)
                               ->dim_order());
        });
        auto tensor = for_blob(view->data(), view->sizes(), view->scalar_type())
                          .strides(view->strides())
                          .dim_order(view->dim_order())
                          .dynamism(aten::TensorShapeDynamism::STATIC)
                          .device(view->device())
                          .make_tensor_ptr();
        dlpack_tensor_ptrs.push_back(std::move(tensor));
        cpp_inputs.emplace_back(dlpack_tensor_ptrs.back());
#endif
      } else if (PyObject_CheckBuffer(python_input.ptr())) {
        buffer_inputs.push_back(
            read_input(storage_->method->method_meta(), i, [&] {
              return std::make_shared<BufferTensor>(
                  python_input, storage_->method->method_meta(), i);
            }));
        const auto& buffer = buffer_inputs.back();
#ifndef USE_ATEN_LIB
        validate_input_device(
            *storage_->method,
            i,
            executorch::aten::Device(executorch::aten::DeviceType::CPU),
            buffer->nbytes());
#endif
        validate_buffer_input(storage_->method->method_meta(), i, *buffer);
        const auto input_meta =
            storage_->method->method_meta().input_tensor_meta(i);
        if (input_meta.ok() && !input_meta->is_memory_planned() &&
            buffer->borrows_data()) {
          borrowed_inputs.emplace_back(i, buffer);
        }
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
            "Input " + std::to_string(i) + " of method '" +
            storage_->method->method_meta().name() +
            "': unsupported python type " + type_str +
            ". Ensure that inputs are passed as a flat list of tensors.");
      }
    }

    executorch::aten::ArrayRef<EValue> input_evalue_list(
        cpp_inputs.data(), cpp_inputs.size());

    Error set_inputs_status = storage_->method->set_inputs(input_evalue_list);
    if (set_inputs_status != Error::Ok) {
      // set_inputs() installs values one at a time and does not roll back a
      // prefix when a later value fails, so refuse to execute or read
      // outputs until a later call installs a full set. Nothing reads the
      // prefix before then, so this call's inputs need not stay alive.
      inputs_failed_ = true;
      outputs_valid_ = false;
      THROW_IF_ERROR(
          set_inputs_status,
          "method->set_inputs() for method '%s' failed with error 0x%" PRIx32,
          storage_->method->method_meta().name(),
          static_cast<uint32_t>(set_inputs_status));
    }
    inputs_failed_ = false;
    inputs_set_ = true;
    // An output can share memory with an input, by value or through memory
    // planning, so it would now show data no execute() produced.
    outputs_valid_ = false;
    buffer_inputs_ = std::move(buffer_inputs);
    buffer_tensor_ptrs_ = std::move(buffer_tensor_ptrs);
    borrowed_inputs_ = std::move(borrowed_inputs);
#ifndef USE_ATEN_LIB
    dlpack_inputs_ = std::move(dlpack_inputs);
    dlpack_tensor_ptrs_ = std::move(dlpack_tensor_ptrs);
    torch_inputs_ = std::move(torch_inputs);
    torch_tensor_ptrs_ = std::move(input_tensors);
#endif
  }

  void execute() {
    ObjectLock lock(mutex_);
    // The output setup below reads inputs through Method::get_input(), which
    // marks them as set and so hides the runtime's own check for unset inputs.
    if (!inputs_set_) {
      throw std::runtime_error(
          "Method '" + std::string(storage_->method->method_meta().name()) +
          "' has no inputs yet. Call set_inputs() before executing.");
    }
    if (inputs_failed_) {
      throw std::runtime_error(
          "The last set_inputs() for method '" +
          std::string(storage_->method->method_meta().name()) +
          "' failed partway, so the method may hold a mix of new and "
          "old inputs. Call set_inputs() again with every input before "
          "executing.");
    }
    outputs_valid_ = false;
    handed_outputs_.clear();
    for (const auto& input : borrowed_inputs_) {
      input.validate(storage_->method->method_meta().name());
    }
    const auto num_outputs = storage_->method->outputs_size();
    allocate_output_storages();
    std::vector<Span<uint8_t>> output_storage_spans(num_outputs);
    for (int i = 0; i < storage_->output_storages.size(); ++i) {
      const auto& device_buffer = storage_->device_output_storages[i];
      output_storage_spans[i] = device_buffer
          ? device_buffer->as_span()
          : Span<uint8_t>(
                storage_->output_storages[i].data(),
                storage_->output_storages[i].size());
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
    setup_output_storage(*storage_->method, output_storage_spans);
#ifndef USE_ATEN_LIB
    synchronize_tensor_devices(*storage_->method);
#endif
    Error execute_status = storage_->method->execute();
    THROW_IF_ERROR(
        execute_status,
        "method->execute() failed with error 0x%" PRIx32,
        static_cast<uint32_t>(execute_status));
#ifndef USE_ATEN_LIB
    synchronize_tensor_devices(*storage_->method);
#endif
    outputs_valid_ = true;
  }

  py::list get_outputs(bool clone_outputs = true) {
    ObjectLock lock(mutex_);
    if (!outputs_valid_) {
      throw std::runtime_error(
          "Method '" + std::string(storage_->method->method_meta().name()) +
          "' has no outputs to return: it has not executed since its last "
          "set_inputs(), or that execution failed. Call execute() first.");
    }
    std::vector<EValue> result(storage_->method->outputs_size());

    Error get_outputs_status = storage_->method->get_outputs(
        result.data(), storage_->method->outputs_size());
    THROW_IF_ERROR(
        get_outputs_status,
        "method->get_outputs() for method '%s' failed with error 0x%" PRIx32,
        storage_->method->method_meta().name(),
        static_cast<uint32_t>(get_outputs_status));

    // Retrieve outputs
    return get_outputs_as_py_list(result, clone_outputs);
  }

  py::list call(const py::object& inputs, bool clone_outputs = true) {
    ObjectLock lock(mutex_);
    set_inputs(inputs);
    execute();
    return get_outputs(clone_outputs);
  }

  py::object get_attribute(const std::string& name) {
    ObjectLock lock(mutex_);
    Result<executorch::aten::Tensor> attr =
        storage_->method->get_attribute(name);
    THROW_IF_ERROR(
        attr.error(),
        "Failed to get attribute '%s' for method '%s', error: 0x:%" PRIx32,
        name.c_str(),
        storage_->method->method_meta().name(),
        static_cast<uint32_t>(attr.error()));
#ifdef USE_ATEN_LIB
    return py::cast(attr.get());
#else
    // No execute() in this call has waited for the work that last wrote it.
    if (!attr->device().is_cpu() && attr->nbytes() != 0) {
      synchronize_device(attr->device().index());
    }
    return portable_tensor_result(
        attr.get(),
        /*clone=*/false,
        borrowed_output_owner(attr.get()));
#endif
  }

  PyMethodMeta method_meta() {
    ObjectLock lock(mutex_);
    return PyMethodMeta(storage_->state, storage_->method->method_meta());
  }

 private:
  std::recursive_mutex mutex_;
  // False until a set_inputs() succeeds. A method without inputs needs none.
  bool inputs_set_ = false;
  // True after a failed set_inputs() left a mix of new and old inputs
  // installed. A method whose inputs were never set keeps its loaded state.
  bool inputs_failed_ = false;
  // True only after the last execute() succeeded.
  bool outputs_valid_ = false;
  // Borrowed results retain native storage without retaining this wrapper's
  // Python input owners, which could point back to those same results.
  struct Storage final {
    std::shared_ptr<ProgramMemory> memory;
    std::shared_ptr<ProgramState> state;
    std::vector<std::vector<uint8_t>> output_storages;
    std::vector<std::shared_ptr<DeviceMemoryBuffer>> device_output_storages;
    std::unique_ptr<Method> method;

    Storage(
        std::shared_ptr<ProgramMemory> memory,
        std::shared_ptr<ProgramState> state,
        std::unique_ptr<Method> method)
        : memory(std::move(memory)),
          state(std::move(state)),
          method(std::move(method)) {}
  };
  std::shared_ptr<Storage> storage_;
  // Keep Python buffer exports and their TensorPtr metadata alive until the
  // next successful set_inputs() call.
  std::vector<std::shared_ptr<BufferTensor>> buffer_inputs_;
  std::vector<TensorPtr> buffer_tensor_ptrs_;
  std::vector<BorrowedInput> borrowed_inputs_;
#ifndef USE_ATEN_LIB
  std::vector<std::shared_ptr<DLPackTensorView>> dlpack_inputs_;
  std::vector<TensorPtr> dlpack_tensor_ptrs_;
  std::vector<std::shared_ptr<TorchTensorView>> torch_inputs_;
  std::vector<TensorPtr> torch_tensor_ptrs_;
#endif
  // What get_outputs() handed back without a copy in this execution.
  std::vector<py::object> handed_outputs_;
  // Per device output buffer, alive while such a result still uses it.
  std::vector<std::weak_ptr<void>> handed_buffers_;
  // Per device output, the buffer a still-alive earlier result uses, kept so
  // it can be reused once that result is dropped.
  struct SpareDeviceOutput final {
    std::shared_ptr<DeviceMemoryBuffer> buffer;
    std::weak_ptr<void> handed;
  };
  std::vector<SpareDeviceOutput> spare_device_outputs_;
#ifndef USE_ATEN_LIB

  struct BorrowedOutputOwner final {
    std::shared_ptr<Storage> storage;
    std::vector<std::shared_ptr<DeviceMemoryBuffer>> device_buffers;
    std::vector<std::shared_ptr<BufferTensor>> buffers;
    std::vector<std::shared_ptr<TorchTensorView>> torch_inputs;
    std::vector<std::shared_ptr<DLPackTensorView>> dlpack_inputs;
  };

  py::object aliased_torch_output(const executorch::aten::Tensor& tensor) {
    if (torch_inputs_.empty() ||
        portable_tensor_output() != PortableTensorOutput::Torch ||
        tensor.nbytes() == 0) {
      return py::none();
    }
    const auto address = reinterpret_cast<uintptr_t>(tensor.const_data_ptr());
    const auto tensor_type =
        py::reinterpret_borrow<py::object>(loaded_torch_module())
            .attr("Tensor");
    for (const auto& input : torch_inputs_) {
      const auto start = reinterpret_cast<uintptr_t>(input->data());
      if (Py_TYPE(input->owner().ptr()) !=
              reinterpret_cast<PyTypeObject*>(tensor_type.ptr()) ||
          tensor.device() != input->device() ||
          tensor.scalar_type() != input->scalar_type() || address < start ||
          address - start > input->nbytes() ||
          tensor.nbytes() > input->nbytes() - (address - start)) {
        continue;
      }
      const auto itemsize = runtime::elementSize(tensor.scalar_type());
      if ((address - start) % itemsize != 0) {
        continue;
      }
      auto source = input->owner().attr("detach")();
      if (source.attr("data_ptr")().cast<uintptr_t>() != start ||
          scalar_type_from_torch_dtype(
              py::str(source.attr("dtype")).cast<std::string>()) !=
              tensor.scalar_type()) {
        throw py::value_error(
            "Input " + std::to_string(input->index()) + " of method '" +
            storage_->method->method_meta().name() +
            "': its storage changed before the output was read; pass it "
            "again");
      }
      const auto offset = source.attr("storage_offset")().cast<int64_t>() +
          static_cast<int64_t>((address - start) / itemsize);
      return source.attr("as_strided")(
          std::vector<int64_t>(tensor.sizes().begin(), tensor.sizes().end()),
          std::vector<int64_t>(
              tensor.strides().begin(), tensor.strides().end()),
          offset);
    }
    return py::none();
  }

  // The result for output `index` when it fills a device buffer these bindings
  // allocated and nothing else uses: the result shares the buffer instead of
  // copying it, and the next execution writes into a new buffer while the
  // result is alive. A buffer that another result already uses, a borrowed
  // one or an earlier output of the same read, is copied instead, so every
  // result stays independent.
  std::optional<py::object> hand_over_device_output(
      size_t index,
      const executorch::aten::Tensor& tensor) {
    const auto slot = device_buffer_slot(tensor);
    if (!slot) {
      return std::nullopt;
    }
    auto& buffer = storage_->device_output_storages[*slot];
    if (buffer.use_count() > 1) {
      return std::nullopt;
    }
    // Results share the buffer through a holder of their own, so
    // handed_buffers_ sees the last of them go even though the slot keeps
    // the buffer.
    const auto holder =
        std::make_shared<std::shared_ptr<DeviceMemoryBuffer>>(buffer);
    const std::shared_ptr<void> memory(holder, buffer->data());
    py::object result;
    if (portable_tensor_output() == PortableTensorOutput::Torch) {
      auto* retained = new std::shared_ptr<void>(memory);
      result = portable_tensor_result(
          tensor,
          /*clone=*/false,
          py::capsule(retained, [](void* pointer) {
            delete static_cast<std::shared_ptr<void>*>(pointer);
          }));
    } else {
      result = py::cast(PyExecuTorchResult::hand_over(tensor, memory));
    }
    handed_buffers_[*slot] = holder;
    handed_outputs_.resize(storage_->method->outputs_size());
    handed_outputs_[index] = result;
    return result;
  }

  std::optional<size_t> device_buffer_slot(
      const executorch::aten::Tensor& tensor) const {
    const auto& buffers = storage_->device_output_storages;
    for (size_t i = 0; i < buffers.size(); ++i) {
      if (buffers[i] && buffers[i]->data() == tensor.const_data_ptr() &&
          buffers[i]->size() == tensor.nbytes()) {
        return i;
      }
    }
    return std::nullopt;
  }

  py::object borrowed_output_owner(const executorch::aten::Tensor& tensor) {
    if (tensor.nbytes() == 0) {
      return py::none();
    }
    auto owner = std::make_unique<BorrowedOutputOwner>();
    owner->storage = storage_;
    const auto address = reinterpret_cast<uintptr_t>(tensor.const_data_ptr());
    const auto aliases = [address](const void* data, size_t nbytes) {
      const auto start = reinterpret_cast<uintptr_t>(data);
      return address != 0 && address >= start && address - start < nbytes;
    };
    bool owns_data = storage_->memory->owns_data(tensor.const_data_ptr());
    for (const auto& buffer : storage_->output_storages) {
      owns_data |= aliases(buffer.data(), buffer.size());
    }
    for (const auto& buffer : storage_->device_output_storages) {
      if (buffer && aliases(buffer->data(), buffer->size())) {
        // The method may write the next execution into a new buffer, so the
        // result keeps the one it reads.
        owner->device_buffers.push_back(buffer);
        owns_data = true;
      }
    }
    if (!owns_data) {
      for (const auto& buffer : buffer_inputs_) {
        if (aliases(buffer->data(), buffer->nbytes())) {
          const auto start = reinterpret_cast<uintptr_t>(buffer->data());
          if (tensor.nbytes() <= buffer->nbytes() - (address - start)) {
            auto backing_owner = buffer->result_backing_owner();
            if (!backing_owner.is_none()) {
              return backing_owner;
            }
          }
          owner->buffers.push_back(buffer);
        }
      }
      for (const auto& input : torch_inputs_) {
        if (aliases(input->data(), input->nbytes())) {
          owner->torch_inputs.push_back(input);
        }
      }
      for (const auto& input : dlpack_inputs_) {
        if (aliases(input->data(), input->nbytes())) {
          const auto start = reinterpret_cast<uintptr_t>(input->data());
          if (tensor.nbytes() <= input->nbytes() - (address - start)) {
            auto backing_owner = input->result_backing_owner();
            if (!backing_owner.is_none()) {
              return backing_owner;
            }
          }
          owner->dlpack_inputs.push_back(input);
        }
      }
    }
    py::capsule capsule(owner.get(), [](void* data) {
      delete static_cast<BorrowedOutputOwner*>(data);
    });
    owner.release();
    return capsule;
  }
#endif

  void allocate_output_storages() {
    const auto num_outputs = storage_->method->outputs_size();
    if (storage_->output_storages.size() == num_outputs) {
      replace_held_device_outputs();
      return;
    }
    // Built in locals and moved in only once every output has its buffer, so
    // an allocation that throws partway leaves nothing behind for the next
    // call to append to.
    std::vector<std::vector<uint8_t>> output_storages;
    std::vector<std::shared_ptr<DeviceMemoryBuffer>> device_output_storages(
        num_outputs);
    // Create a buffer for each output tensor. Memory planned outputs and non
    // tensor outputs get an empty buffer in this list which is ignored later.
    output_storages.reserve(num_outputs);
    auto meta = storage_->method->method_meta();
    for (size_t i = 0; i < num_outputs; ++i) {
      auto output_type = meta.output_tag(i);
      THROW_IF_ERROR(
          output_type.error(), "Failed to get output type for output %zu", i);
      if (output_type.get() != Tag::Tensor) {
        // Skip allocating storage for non-tensor outputs.
        output_storages.emplace_back();
        continue;
      }
      const auto& output_tensor_meta =
          storage_->method->method_meta().output_tensor_meta(i);
      THROW_IF_ERROR(
          output_tensor_meta.error(),
          "Failed to get output tensor meta for output %zu",
          i);
      if (output_tensor_meta.get().is_memory_planned()) {
        // Skip allocating storage for planned memory outputs.
        output_storages.emplace_back();
        continue;
      }
      const auto& output = storage_->method->get_output(i);
      bool is_input = false;
      for (size_t j = 0; j < storage_->method->inputs_size(); ++j) {
        is_input |= &output == &storage_->method->get_input(j);
      }
      if (is_input) {
        output_storages.emplace_back();
        continue;
      }
      // Allocate storage for the output tensor.
      const size_t output_size = output_tensor_meta.get().nbytes();
#ifndef USE_ATEN_LIB
      // The delegate and the output clone treat a device-tagged output as
      // device memory. A host buffer there only works on a device that can
      // read pageable host memory, and fails everywhere else. An empty output
      // needs no memory on any device, and whether an allocator accepts a zero
      // size is up to the allocator, so it keeps the empty host buffer below,
      // which setup_output_storage skips.
      const auto device = storage_->method->get_output(i).toTensor().device();
      if (!device.is_cpu() && output_size > 0) {
        device_output_storages[i] =
            allocate_device_output(i, output_size, device);
        output_storages.emplace_back();
        continue;
      }
#endif
      output_storages.emplace_back(output_size);
    }
    storage_->output_storages = std::move(output_storages);
    storage_->device_output_storages = std::move(device_output_storages);
    handed_buffers_.assign(num_outputs, std::weak_ptr<void>());
    spare_device_outputs_.assign(num_outputs, SpareDeviceOutput());
  }

  // A handed-over result still uses its buffer, so the method writes into
  // its spare instead, and the two take turns. Borrowed results keep their
  // buffer alive themselves, as they always could see the next execution's
  // values.
  void replace_held_device_outputs() {
#ifndef USE_ATEN_LIB
    for (size_t i = 0; i < storage_->device_output_storages.size(); ++i) {
      auto& buffer = storage_->device_output_storages[i];
      if (!buffer || handed_buffers_[i].expired()) {
        continue;
      }
      const auto device = storage_->method->get_output(i).toTensor().device();
      auto& spare = spare_device_outputs_[i];
      if (spare.buffer && spare.handed.expired()) {
        std::swap(buffer, spare.buffer);
        std::swap(handed_buffers_[i], spare.handed);
      } else {
        // Allocated into a local first, so a failed allocation leaves the slot
        // and the spare untouched.
        auto fresh = allocate_device_output(i, buffer->size(), device);
        spare = {std::move(buffer), handed_buffers_[i]};
        buffer = std::move(fresh);
        handed_buffers_[i].reset();
      }
    }
#endif
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
        // Within one execution, repeated default reads return the same handed
        // over object, so a caller sees stable identity and the buffer is not
        // handed over twice. A clone_outputs=False read is a separate borrowed
        // view and never returns that cached object.
        if (clone_outputs && i < handed_outputs_.size() && handed_outputs_[i]) {
          list[i] = handed_outputs_[i];
          continue;
        }
        if (clone_outputs) {
          if (const auto handed = hand_over_device_output(i, v.toTensor())) {
            list[i] = *handed;
            continue;
          }
        }
        if (!clone_outputs) {
          auto alias = aliased_torch_output(v.toTensor());
          if (!alias.is_none()) {
            list[i] = std::move(alias);
            continue;
          }
        }
        list[i] = portable_tensor_result(
            v.toTensor(),
            clone_outputs,
            clone_outputs && v.toTensor().device().is_cpu()
                ? py::none()
                : borrowed_output_owner(v.toTensor()));
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
      std::optional<const std::string> data_path = std::nullopt,
      std::optional<py::bytes> program_buffer = std::nullopt,
      const py::dict& backend_options = py::dict())
      : state_(load_program(
            std::move(loader),
            program_verification,
            data_path,
            std::move(program_buffer),
            backend_options)),
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
      std::optional<const std::string> data_path = std::nullopt,
      const py::dict& backend_options = py::dict()) {
    std::unique_ptr<DataLoader> loader = loader_from_buffer(
        buffer.cast<std::string_view>().data(), py::len(buffer));
    return std::make_unique<PyProgram>(
        std::move(loader),
        enable_etdump ? std::make_unique<torch::executor::ETDumpGen>()
                      : nullptr,
        debug_buffer_size,
        program_verification,
        data_path,
        buffer,
        backend_options);
  }

  static std::unique_ptr<PyProgram> load_from_file(
      const std::string& path,
      bool enable_etdump,
      size_t debug_buffer_size,
      Program::Verification program_verification =
          Program::Verification::Minimal,
      std::optional<const std::string> data_path = std::nullopt,
      const py::dict& backend_options = py::dict()) {
    std::unique_ptr<DataLoader> loader = loader_from_file(path);
    return std::make_unique<PyProgram>(
        std::move(loader),
        enable_etdump ? std::make_unique<torch::executor::ETDumpGen>()
                      : nullptr,
        debug_buffer_size,
        program_verification,
        data_path,
        std::nullopt,
        backend_options);
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
        state_->data_map_.get(),
        &state_->backend_options_);
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

void set_option(const std::string& backend_name, const py::dict& options) {
  auto backend_options = to_backend_options(options);
  const Error error = ::executorch::ET_RUNTIME_NAMESPACE::set_option(
      backend_name.c_str(),
      Span<BackendOption>(backend_options.data(), backend_options.size()));
  THROW_IF_ERROR(
      error,
      "Failed to set options for backend %s, error: 0x%" PRIx32,
      backend_name.c_str(),
      static_cast<uint32_t>(error));
}

py::dict get_option(const std::string& backend_name, const py::dict& options) {
  auto backend_options = to_backend_options(options);
  const Error error = ::executorch::ET_RUNTIME_NAMESPACE::get_option(
      backend_name.c_str(),
      Span<BackendOption>(backend_options.data(), backend_options.size()));
  THROW_IF_ERROR(
      error,
      "Failed to get options for backend %s, error: 0x%" PRIx32,
      backend_name.c_str(),
      static_cast<uint32_t>(error));
  py::dict result;
  for (const auto& option : backend_options) {
    if (const auto* flag = std::get_if<bool>(&option.value)) {
      result[option.key] = *flag;
    } else if (const auto* number = std::get_if<int>(&option.value)) {
      result[option.key] = *number;
    } else {
      result[option.key] =
          std::get<std::array<char, kMaxOptionValueLength>>(option.value)
              .data();
    }
  }
  return result;
}

} // namespace

PYBIND11_MODULE(EXECUTORCH_PYTHON_MODULE_NAME, m) {
  // scoped_ostream_redirect mutates process-wide stream buffers and is not
  // thread-safe when Python operations inside a binding release the GIL.
  auto call_guard = py::call_guard<>();

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
      m,
      "ExecuTorchResult",
      py::buffer_protocol(),
      py::module_local(),
      "A tensor result of an ExecuTorch method, returned when PyTorch is not "
      "used. Read a CPU result through the buffer protocol, NumPy, or DLPack, "
      "and a CUDA result through DLPack or to_host().\n\n"
      ".. warning::\n\n"
      "    This API is experimental and subject to change without notice.")
      .def_property_readonly(
          "shape", &PyExecuTorchResult::shape, "Size of each dimension.")
      .def_property_readonly(
          "strides",
          &PyExecuTorchResult::strides,
          "Byte step of each dimension.")
      .def_property_readonly(
          "dtype",
          &PyExecuTorchResult::dtype,
          "NumPy dtype. Raises for dtypes NumPy lacks, such as bfloat16.")
      .def_property_readonly(
          "dtype_name",
          &PyExecuTorchResult::torch_dtype_name,
          "Name of the dtype, such as \"bfloat16\", including dtypes NumPy "
          "lacks.")
      .def_property_readonly(
          "nbytes", &PyExecuTorchResult::nbytes, "Size of the data in bytes.")
      .def(
          "__array__",
          [](const std::shared_ptr<PyExecuTorchResult>& result,
             const py::object& dtype,
             const py::object& copy) {
            py::array array = result->numpy_array(py::cast(result));
            const bool same_dtype = dtype.is_none() ||
                array.dtype().equal(py::dtype::from_args(dtype));
            if (!copy.is_none() && !copy.cast<bool>() && !same_dtype) {
              throw py::value_error(
                  "A dtype conversion requires a copy of the ExecuTorch result");
            }
            if (!same_dtype) {
              return array.attr("astype")(dtype);
            }
            return !copy.is_none() && copy.cast<bool>()
                ? array.attr("copy")(py::arg("order") = "K")
                : py::object(std::move(array));
          },
          py::arg("dtype") = py::none(),
          py::arg("copy") = py::none())
      .def_buffer(&PyExecuTorchResult::buffer)
      .def(
          "__dlpack__",
          &PyExecuTorchResult::to_dlpack,
          py::arg("stream") = py::none(),
          py::kw_only(),
          py::arg("max_version") = py::none(),
          py::arg("dl_device") = py::none(),
          py::arg("copy") = py::none())
      .def("__dlpack_device__", &PyExecuTorchResult::dlpack_device)
      .def(
          "to_host",
          &PyExecuTorchResult::to_host,
          "Return a copy of this result in CPU memory.")
      .def_static(
          "from_host",
          [](const py::object& source, int device_index) {
            if (py::isinstance<PyExecuTorchResult>(source)) {
              return source.cast<std::shared_ptr<PyExecuTorchResult>>()
                  ->to_device(device_index);
            }
            const BufferTensor host(source);
            return PyExecuTorchResult::from_host(
                host.data(),
                std::vector<executorch::aten::SizesType>(
                    host.sizes().begin(), host.sizes().end()),
                std::vector<executorch::aten::StridesType>(
                    host.strides().begin(), host.strides().end()),
                host.scalar_type(),
                device_index);
          },
          py::arg("source"),
          py::arg("device_index") = 0,
          "Copy a dense CPU buffer, or a CPU ExecuTorchResult, to a CUDA "
          "device, for use as a method input. Requires bindings built with "
          "CUDA support.")
      .def("__repr__", &PyExecuTorchResult::repr);
#ifndef USE_ATEN_LIB
  py::class_<
      PyExecuTorchResultFlatBuffer,
      std::shared_ptr<PyExecuTorchResultFlatBuffer>>(
      m,
      "_ExecuTorchResultFlatBuffer",
      py::buffer_protocol(),
      py::module_local())
      .def_buffer(&PyExecuTorchResultFlatBuffer::buffer);
#endif

  // Bind the verification enum to python.
  py::enum_<Program::Verification>(m, "Verification", py::module_local())
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
  m.def(
      "_set_option",
      &set_option,
      py::arg("backend_name"),
      py::arg("options"),
      call_guard);
  m.def(
      "_get_option",
      &get_option,
      py::arg("backend_name"),
      py::arg("options"),
      call_guard);
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

  py::class_<PyModule>(m, "ExecuTorchModule", py::module_local())
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

  py::class_<PyBundledModule>(m, "BundledModule", py::module_local())
      .def(
          "verify_result_with_bundled_expected_output",
          &PyBundledModule::verify_result_with_bundled_expected_output,
          py::arg("method_name"),
          py::arg("testset_idx"),
          py::arg("rtol") = 1e-5,
          py::arg("atol") = 1e-8,
          call_guard);

  py::class_<PyTensorInfo>(m, "TensorInfo", py::module_local())
      .def("sizes", &PyTensorInfo::sizes, call_guard)
      .def("dtype", &PyTensorInfo::dtype, call_guard)
      .def("is_memory_planned", &PyTensorInfo::is_memory_planned, call_guard)
      .def("nbytes", &PyTensorInfo::nbytes, call_guard)
      .def("__repr__", &PyTensorInfo::repr, call_guard);
  py::class_<PyMethodMeta>(m, "MethodMeta", py::module_local())
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
      py::arg("backend_options") = py::dict(),
      call_guard);
  m.def(
      "_load_program_from_buffer",
      &PyProgram::load_from_buffer,
      py::arg("buffer"),
      py::arg("enable_etdump") = false,
      py::arg("debug_buffer_size") = 0,
      py::arg("program_verification") = Program::Verification::Minimal,
      py::arg("data_path") = std::nullopt,
      py::arg("backend_options") = py::dict(),
      call_guard);
  py::class_<PyProgram>(m, "ExecuTorchProgram", py::module_local())
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
  py::class_<PyMethod>(m, "ExecuTorchMethod", py::module_local())
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

// Forward runtime logs to Python without mutating std::cerr's process-wide
// stream buffer.
void emit_log_message(
    ET_UNUSED et_timestamp_t timestamp,
    ET_UNUSED et_pal_log_level_t level,
    const char* filename,
    ET_UNUSED const char* function,
    size_t line,
    const char* message,
    size_t length) {
  const std::string formatted_message = "[" + std::string(filename) + ":" +
      std::to_string(line) + "] " + std::string(message, length) + "\n";
  if (Py_IsInitialized() && PyGILState_Check()) {
    try {
      py::gil_scoped_acquire gil;
      py::module_::import("sys").attr("stderr").attr("write")(
          formatted_message);
      return;
    } catch (py::error_already_set& error) {
      error.clear();
    }
  }
  std::cerr << formatted_message << std::flush;
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
