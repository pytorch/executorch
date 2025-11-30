/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <executorch/devtools/bundled_program/bundled_program.h>
#include <executorch/devtools/bundled_program/schema/bundled_program_schema_generated.h>
#include <executorch/devtools/etdump/etdump_flatcc.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/data_loader/mmap_data_loader.h>
#include <executorch/extension/memory_allocator/malloc_memory_allocator.h>
#include <executorch/extension/module/bundled_module.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/extension/threadpool/threadpool.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>

#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/core/functional.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>

#ifndef USE_ATEN_LIB
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <executorch/extension/aten_util/aten_bridge.h>
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
using executorch::BUNDLED_PROGRAM_NAMESPACE::verify_method_outputs;
using ::executorch::ET_RUNTIME_NAMESPACE::BackendInterface;
using ::executorch::ET_RUNTIME_NAMESPACE::get_backend_class;
using ::executorch::ET_RUNTIME_NAMESPACE::get_backend_name;
using ::executorch::ET_RUNTIME_NAMESPACE::get_num_registered_backends;
using ::executorch::ET_RUNTIME_NAMESPACE::get_registered_kernels;
using ::executorch::ET_RUNTIME_NAMESPACE::Kernel;
using ::executorch::ET_RUNTIME_NAMESPACE::Method;
using ::executorch::ET_RUNTIME_NAMESPACE::Program;
using ::executorch::extension::BufferDataLoader;
using ::executorch::extension::MallocMemoryAllocator;
using ::executorch::extension::MmapDataLoader;
using ::executorch::extension::ET_BUNDLED_MODULE_NAMESPACE::BundledModule;
using ::executorch::runtime::ArrayRef;
using ::executorch::runtime::DataLoader;
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

#ifndef USE_ATEN_LIB
using ::executorch::extension::alias_attensor_to_etensor;
using ::executorch::extension::alias_etensor_to_attensor;
using ::executorch::extension::torch_to_executorch_scalar_type;
#endif // !USE_ATEN_LIB

namespace executorch {
namespace extension {
namespace pybindings {

namespace {

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
      if (clone_outputs) {
        list[i] = py::cast(alias_attensor_to_etensor(v.toTensor()).clone());
      } else {
        list[i] = py::cast(alias_attensor_to_etensor(v.toTensor()));
      }
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

  explicit ProgramState(
      std::unique_ptr<DataLoader> loader,
      std::unique_ptr<Program> program)
      : loader_(std::move(loader)), program_(std::move(program)) {}
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

  py::list run_method(
      const std::string& method_name,
      const py::sequence& inputs,
      bool clone_outputs = true) {
    const auto inputs_size = py::len(inputs);
    std::vector<EValue> cpp_inputs;
    cpp_inputs.reserve(inputs_size);

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
      if (type_str == "<class 'torch.Tensor'>") {
        auto at_tensor = python_input.cast<at::Tensor>();

#ifdef USE_ATEN_LIB
        EValue evalue(at_tensor);
#else
        // convert at::Tensor to torch::executor::Tensor
        auto type =
            torch_to_executorch_scalar_type(at_tensor.options().dtype());
        size_t dim = at_tensor.dim();
        // cant directly alias at::Tensor sizes and strides due to int64 vs
        // int32 typing conflict
        input_sizes.emplace_back(
            at_tensor.sizes().begin(), at_tensor.sizes().end());
        input_strides.emplace_back(
            at_tensor.strides().begin(), at_tensor.strides().end());

        // Only works for MemoryFormat::Contiguous or MemoryFormat::ChannelsLast
        // inputs
        std::vector<torch::executor::Tensor::DimOrderType> dim_order;
        if (at_tensor.is_contiguous()) {
          for (size_t cur_dim = 0; cur_dim < dim; cur_dim++) {
            dim_order.push_back(cur_dim);
          }
        } else if (
            at_tensor.is_contiguous(at::MemoryFormat::ChannelsLast) &&
            at_tensor.dim() == 4) {
          dim_order = decltype(dim_order)({0, 2, 3, 1});
        } else {
          auto error_msg = "Input " + std::to_string(i) + "for method " +
              method_name + " should be contiguous or channels-last.";
          throw std::runtime_error(error_msg);
        }
        input_dim_order.push_back(std::move(dim_order));
        input_tensors.emplace_back(
            type,
            dim,
            input_sizes.back().data(),
            nullptr,
            input_dim_order.back().data(),
            input_strides.back().data());

        torch::executor::Tensor temp =
            torch::executor::Tensor(&input_tensors.back());
        alias_etensor_to_attensor(at_tensor, temp);
        EValue evalue(temp);
#endif

        cpp_inputs.push_back(evalue);
      } else if (py::isinstance<py::none>(python_input)) {
        cpp_inputs.push_back(EValue());
      } else if (py::isinstance<py::bool_>(python_input)) {
        cpp_inputs.push_back(EValue(py::cast<bool>(python_input)));
      } else if (py::isinstance<py::int_>(python_input)) {
        cpp_inputs.push_back(EValue(py::cast<int64_t>(python_input)));
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

  py::list forward(const py::sequence& inputs, bool clone_outputs = true) {
    return run_method("forward", inputs, clone_outputs);
  }

  py::list forward_single_input(
      const torch::Tensor& inputTensor,
      bool clone_outputs = true) {
    py::list py_list;
    py_list.append(py::cast(inputTensor));
    return run_method("forward", py_list, clone_outputs);
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
    Program::Verification program_verification) {
  Result<Program> res = Program::load(loader.get(), program_verification);
  THROW_IF_ERROR(
      res.error(),
      "Failed to load program, error: 0x:%" PRIx32,
      static_cast<uint32_t>(res.error()));
  return std::make_shared<ProgramState>(
      std::move(loader), std::make_unique<Program>(std::move(res.get())));
}

/// A wrapper/util class for executorch memory allocations/manager.
class ProgramMemory {
 public:
  explicit ProgramMemory(std::vector<std::vector<uint8_t>>&& non_const_buffers)
      : runtime_allocator_(),
        non_const_buffers_(std::move(non_const_buffers)),
        non_const_spans_(create_non_const_spans()),
        non_const_allocator_(
            {non_const_spans_.data(), non_const_spans_.size()}),
        mem_manager_(
            &const_allocator_,
            &non_const_allocator_,
            &runtime_allocator_,
            &temp_allocator_) {}

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

  std::vector<std::vector<uint8_t>> non_const_buffers_;

  std::vector<Span<uint8_t>> non_const_spans_;

  HierarchicalAllocator non_const_allocator_;

  MemoryManager mem_manager_;

  std::vector<Span<uint8_t>> create_non_const_spans() {
    std::vector<Span<uint8_t>> result;
    for (size_t i = 0; i < non_const_buffers_.size(); i++) {
      result.push_back(
          {non_const_buffers_[i].data(), non_const_buffers_[i].size()});
    }
    return result;
  }
};

struct PyMethod final {
  explicit PyMethod(
      std::shared_ptr<ProgramMemory> memory,
      std::shared_ptr<ProgramState> state,
      std::unique_ptr<Method> method)
      : memory_(std::move(memory)),
        state_(std::move(state)),
        method_(std::move(method)) {}

  void set_inputs(const py::sequence& inputs) {
    const auto inputs_size = py::len(inputs);
    std::vector<EValue> cpp_inputs;
    cpp_inputs.reserve(inputs_size);

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
      if (type_str == "<class 'torch.Tensor'>") {
        auto at_tensor = python_input.cast<at::Tensor>();

#ifdef USE_ATEN_LIB
        EValue evalue(at_tensor);
#else
        // convert at::Tensor to torch::executor::Tensor
        auto type =
            torch_to_executorch_scalar_type(at_tensor.options().dtype());
        size_t dim = at_tensor.dim();
        // cant directly alias at::Tensor sizes and strides due to int64 vs
        // int32 typing conflict
        std::vector<int> sizes(
            at_tensor.sizes().begin(), at_tensor.sizes().end());
        std::vector<int> strides(
            at_tensor.strides().begin(), at_tensor.strides().end());

        // Only works for MemoryFormat::Contiguous or MemoryFormat::ChannelsLast
        // inputs
        std::vector<torch::executor::Tensor::DimOrderType> dim_order;
        if (at_tensor.is_contiguous()) {
          for (size_t cur_dim = 0; cur_dim < dim; cur_dim++) {
            dim_order.push_back(cur_dim);
          }
        } else if (
            at_tensor.is_contiguous(at::MemoryFormat::ChannelsLast) &&
            at_tensor.dim() == 4) {
          dim_order = decltype(dim_order)({0, 2, 3, 1});
        } else {
          auto error_msg = "Input " + std::to_string(i) + "for method " +
              method_->method_meta().name() +
              " should be contiguous or channels-last.";
          throw std::runtime_error(error_msg);
        }
        TensorPtr tensor =
            for_blob(at_tensor.data_ptr(), std::move(sizes), type)
                .strides(std::move(strides))
                .dim_order(std::move(dim_order))
                .dynamism(aten::TensorShapeDynamism::STATIC)
                .make_tensor_ptr();
        input_tensors.push_back(tensor);
        EValue evalue(input_tensors.back());
#endif

        cpp_inputs.push_back(evalue);
      } else if (py::isinstance<py::none>(python_input)) {
        cpp_inputs.push_back(EValue());
      } else if (py::isinstance<py::bool_>(python_input)) {
        cpp_inputs.push_back(EValue(py::cast<bool>(python_input)));
      } else if (py::isinstance<py::int_>(python_input)) {
        cpp_inputs.push_back(EValue(py::cast<int64_t>(python_input)));
      } else {
        throw std::runtime_error(
            "Unsupported python type " + type_str +
            ". Ensure that inputs are passed as a flat list of tensors.");
      }
    }

    executorch::aten::ArrayRef<EValue> input_evalue_list(
        cpp_inputs.data(), cpp_inputs.size());

    Error set_inputs_status = method_->set_inputs(input_evalue_list);
    THROW_IF_ERROR(
        set_inputs_status,
        "method->set_inputs() for method '%s' failed with error 0x%" PRIx32,
        method_->method_meta().name(),
        static_cast<uint32_t>(set_inputs_status));
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

  py::list call(const py::sequence& inputs, bool clone_outputs = true) {
    set_inputs(inputs);
    execute();
    return get_outputs(clone_outputs);
  }

  py::list call_single_input(
      const torch::Tensor& inputTensor,
      bool clone_outputs = true) {
    py::list py_list;
    py_list.append(py::cast(inputTensor));
    return call(py_list, clone_outputs);
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
    return py::cast(alias_attensor_to_etensor(attr.get()));
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
        if (clone_outputs) {
          list[i] = py::cast(alias_attensor_to_etensor(v.toTensor()).clone());
        } else {
          list[i] = py::cast(alias_attensor_to_etensor(v.toTensor()));
        }
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
          Program::Verification::Minimal)
      : state_(load_program(std::move(loader), program_verification)),
        event_tracer_(std::move(tracer)),
        debug_buffer_size_(debug_buffer_size) {
    // Figure out the size of each non_const layer we need to support every
    // method in the program. Map will be easier to use than a list because we
    // dont know how many non_const arenas there will be
    std::map<size_t, int64_t> non_const_buffer_sizes;
    for (size_t i = 0; i < state_->program_->num_methods(); ++i) {
      auto name = state_->program_->get_method_name(i).get();
      auto method_meta = state_->program_->method_meta(name).get();
      for (size_t j = 0; j < method_meta.num_non_const_buffers(); j++) {
        int64_t buffer_size = method_meta.non_const_buffer_size(j).get();
        if (non_const_buffer_sizes.find(j) == non_const_buffer_sizes.end()) {
          non_const_buffer_sizes.insert({j, buffer_size});
        } else {
          non_const_buffer_sizes[j] =
              std::max(non_const_buffer_sizes[j], buffer_size);
        }
      }
    }

    // Allocate the arenas. Using vector because we need to remember the size as
    // well, so vector is easier then unique_ptr.
    std::vector<std::vector<uint8_t>> non_const_buffers_;
    for (std::map<size_t, int64_t>::iterator i = non_const_buffer_sizes.begin();
         i != non_const_buffer_sizes.end();
         i++) {
      non_const_buffers_.push_back(std::vector<uint8_t>(i->second));
    }

    memory_ = std::make_shared<ProgramMemory>(std::move(non_const_buffers_));
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
          Program::Verification::Minimal) {
    std::unique_ptr<DataLoader> loader = loader_from_buffer(
        buffer.cast<std::string_view>().data(), py::len(buffer));
    return std::make_unique<PyProgram>(
        std::move(loader),
        enable_etdump ? std::make_unique<torch::executor::ETDumpGen>()
                      : nullptr,
        debug_buffer_size,
        program_verification);
  }

  static std::unique_ptr<PyProgram> load_from_file(
      const std::string& path,
      bool enable_etdump,
      size_t debug_buffer_size,
      Program::Verification program_verification =
          Program::Verification::Minimal) {
    std::unique_ptr<DataLoader> loader = loader_from_file(path);
    return std::make_unique<PyProgram>(
        std::move(loader),
        enable_etdump ? std::make_unique<torch::executor::ETDumpGen>()
                      : nullptr,
        debug_buffer_size,
        program_verification);
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
    Result<Method> res = state_->program_->load_method(
        method_name.c_str(), memory_->mem_manager(), event_tracer_.get());
    THROW_IF_ERROR(
        res.error(),
        "Failed to load method %s, error: 0x:%" PRIx32,
        method_name.c_str(),
        static_cast<uint32_t>(res.error()));
    return std::make_unique<PyMethod>(
        memory_, state_, std::make_unique<Method>(std::move(res.get())));
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
          call_guard)
      .def(
          "__call__",
          &PyModule::forward_single_input,
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
      call_guard);
  m.def(
      "_load_program_from_buffer",
      &PyProgram::load_from_buffer,
      py::arg("buffer"),
      py::arg("enable_etdump") = false,
      py::arg("debug_buffer_size") = 0,
      py::arg("program_verification") = Program::Verification::Minimal,
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
          "call",
          &PyMethod::call_single_input,
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
          "__call__",
          &PyMethod::call_single_input,
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
