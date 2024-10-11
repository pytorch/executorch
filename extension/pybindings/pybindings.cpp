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
#include <unordered_map>

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <executorch/devtools/bundled_program/bundled_program.h>
#include <executorch/devtools/bundled_program/schema/bundled_program_schema_generated.h>
#include <executorch/devtools/etdump/etdump_flatcc.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/data_loader/mmap_data_loader.h>
#include <executorch/extension/memory_allocator/malloc_memory_allocator.h>
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

// Our logs work by writing to stderr. By default this is done through fprintf
// (as defined in posix.cpp) which then does not show up in python environments.
// Here we override the pal to use std::cerr which can be properly redirected by
// scoped_estream_redirect.
void et_pal_emit_log_message(
    et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    ET_UNUSED const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  std::cerr << "[" << filename << ":" << line << "] " << message << std::endl;
}

namespace py = pybind11;
using executorch::bundled_program::verify_method_outputs;
using ::executorch::extension::BufferDataLoader;
using ::executorch::extension::MallocMemoryAllocator;
using ::executorch::extension::MmapDataLoader;
using ::executorch::runtime::ArrayRef;
using ::executorch::runtime::DataLoader;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::EventTracerDebugLogLevel;
using ::executorch::runtime::get_registered_kernels;
using ::executorch::runtime::HierarchicalAllocator;
using ::executorch::runtime::Kernel;
using ::executorch::runtime::MemoryAllocator;
using ::executorch::runtime::MemoryManager;
using ::executorch::runtime::Method;
using ::executorch::runtime::prof_result_t;
using ::executorch::runtime::Program;
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
        Error(),
        "number of output storages %zu does not match number of outputs %zu",
        output_storages.size(),
        method.outputs_size());
  }
  for (size_t i = 0; i < output_storages.size(); ++i) {
    if (output_storages[i].size() == 0) {
      // Skip empty output storages, this would happen for non-tensor outputs.
      continue;
    }
    Error output_status = method.set_output_data_ptr(
        output_storages[i].data(), output_storages[i].size(), i);
    // InvalidState can be the status if outputs are already memory planned.
    // That's fine and we don't need to alert the user to that error.
    if (output_status != Error::Ok && output_status != Error::InvalidState) {
      ET_LOG(
          Error,
          "Cannot set_output_data_ptr(): 0x%" PRIx32,
          static_cast<uint32_t>(output_status));
    }
  }
}

class Module final {
 public:
  explicit Module(
      std::unique_ptr<DataLoader> loader,
      std::unique_ptr<ETDumpGen> tracer = nullptr,
      size_t debug_buffer_size = 0,
      Program::Verification program_verification =
          Program::Verification::InternalConsistency)
      : loader_(std::move(loader)),
        event_tracer_(std::move(tracer)),
        debug_buffer_size_(debug_buffer_size) {
    ::executorch::runtime::runtime_init();
    Result<Program> program =
        Program::load(loader_.get(), program_verification);
    THROW_IF_ERROR(
        program.error(),
        "loading program failed with error: 0x%" PRIx32,
        static_cast<uint32_t>(program.error()));
    program_ = std::make_unique<Program>(std::move(program.get()));

    // Figure out the size of each non_const layer we need to support every
    // method in the program. Map will be easier to use than a list because we
    // dont know how many non_const arenas there will be
    std::map<size_t, int64_t> non_const_buffer_sizes;
    for (size_t i = 0; i < program_->num_methods(); ++i) {
      auto name = program_->get_method_name(i).get();
      auto method_meta = program_->method_meta(name).get();
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

    memory_ = std::make_unique<Memory>(std::move(non_const_buffers_));
    if (event_tracer_ && debug_buffer_size > 0) {
      // If a debug buffer was requested for the ETDump, allocate it and make
      // sure its lifetime is as long as the event_tracer.
      debug_buffer_ = std::make_unique<uint8_t[]>(debug_buffer_size);
      event_tracer_->set_debug_buffer(get_etdump_debug_buffer());
      event_tracer_->set_event_tracer_debug_level(
          EventTracerDebugLogLevel::kIntermediateOutputs);
    }

    // Load methods
    for (size_t i = 0; i < program_->num_methods(); ++i) {
      auto name = program_->get_method_name(i).get();
      // It's safe to use the same memory manager for all modules because
      // we can guarantee that only one will be executing at a time.
      // Everything in this module runs on a single thread.
      Result<Method> method = program_->load_method(
          name, memory_->mem_manager(), event_tracer_.get());
      THROW_IF_ERROR(
          method.error(),
          "loading method %s failed with error 0x%" PRIx32,
          name,
          static_cast<uint32_t>(method.error()));
      methods_.insert(
          {std::string(name),
           std::make_unique<Method>(std::move(method.get()))});
    }
  }

  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;
  Module(Module&&) = default;
  Module& operator=(Module&&) = default;

  /// Executes the specified method on the provided inputs and returns its
  /// outputs.
  std::vector<EValue> run_method(
      const std::string& method_name,
      const std::vector<EValue>& args,
      const std::optional<std::vector<Span<uint8_t>>>& output_storages =
          std::nullopt) {
    auto& method = get_method(method_name);
    exec_aten::ArrayRef<EValue> input_evalue_list(args.data(), args.size());

    Error set_inputs_status = method.set_inputs(input_evalue_list);
    THROW_IF_ERROR(
        set_inputs_status,
        "method->set_inputs() for method '%s' failed with error 0x%" PRIx32,
        method_name.c_str(),
        static_cast<uint32_t>(set_inputs_status));

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
    if (output_storages) {
      setup_output_storage(method, *output_storages);
    }
    Error execute_status = method.execute();
    THROW_IF_ERROR(
        execute_status,
        "method->execute() failed with error 0x%" PRIx32,
        static_cast<uint32_t>(execute_status));
    // process outputs
    return get_outputs(method_name);
  }

  std::vector<EValue> get_outputs(const std::string& method_name) {
    auto& method = methods_[method_name];
    std::vector<EValue> result(method->outputs_size());

    Error get_outputs_status =
        method->get_outputs(result.data(), method->outputs_size());
    THROW_IF_ERROR(
        get_outputs_status,
        "method->get_outputs() for method '%s' failed with error 0x%" PRIx32,
        method_name.c_str(),
        static_cast<uint32_t>(get_outputs_status));

    return result;
  }

  Method& get_method(const std::string& method_name) {
    if (methods_.count(method_name) == 0) {
      THROW_IF_ERROR(
          Error::InvalidArgument,
          "no such method in program: %s",
          method_name.c_str());
    }
    return *methods_[method_name].get();
  }

  /// Returns the names of all methods in the program.
  std::vector<std::string> method_names() const {
    std::vector<std::string> names;
    for (const auto& method : methods_) {
      names.push_back(method.first);
    }
    return names;
  }

  bool has_etdump() {
    return static_cast<bool>(event_tracer_);
  }

  ETDumpGen& etdump() {
    return *event_tracer_;
  }

  bool has_etdump_debug_buffer() const {
    return static_cast<bool>(debug_buffer_);
  }

  Span<uint8_t> get_etdump_debug_buffer() {
    return Span<uint8_t>(debug_buffer_.get(), debug_buffer_size_);
  }

 private:
  /// A wrapper/util class for executorch memory allocations/manager.
  class Memory {
   public:
    explicit Memory(std::vector<std::vector<uint8_t>>&& non_const_buffers)
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

    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;

   private:
    MemoryAllocator const_allocator_{MemoryAllocator(0, nullptr)};

    MallocMemoryAllocator runtime_allocator_;

    MemoryAllocator temp_allocator_{MemoryAllocator(0, nullptr)};

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

  std::unique_ptr<Memory> memory_;
  std::unique_ptr<DataLoader> loader_; // program_ points to this.
  std::unique_ptr<const Program> program_; // methods_ entries points to this.
  std::unordered_map<std::string, std::unique_ptr<Method>> methods_;
  std::unique_ptr<ETDumpGen> event_tracer_;
  std::unique_ptr<uint8_t[]> debug_buffer_;
  size_t debug_buffer_size_;
};

inline std::unique_ptr<Module> load_module_from_buffer(
    const void* ptr,
    size_t ptr_len,
    bool enable_etdump,
    size_t debug_buffer_size,
    Program::Verification program_verification) {
  EXECUTORCH_SCOPE_PROF("load_module_from_buffer");
  auto loader = std::make_unique<BufferDataLoader>(ptr, ptr_len);
  return std::make_unique<Module>(
      std::move(loader),
      enable_etdump ? std::make_unique<torch::executor::ETDumpGen>() : nullptr,
      debug_buffer_size,
      program_verification);
}

inline std::unique_ptr<Module> load_module_from_file(
    const std::string& path,
    bool enable_etdump,
    size_t debug_buffer_size,
    Program::Verification program_verification) {
  EXECUTORCH_SCOPE_PROF("load_module_from_file");

  Result<MmapDataLoader> res = MmapDataLoader::from(
      path.c_str(), MmapDataLoader::MlockConfig::UseMlockIgnoreErrors);
  THROW_IF_ERROR(
      res.error(),
      "Failed to create MmapDataLoader from file %s, error: 0x:%" PRIx32,
      path.c_str(),
      static_cast<uint32_t>(res.error()));

  auto loader = std::make_unique<MmapDataLoader>(std::move(res.get()));
  return std::make_unique<Module>(
      std::move(loader),
      enable_etdump ? std::make_unique<torch::executor::ETDumpGen>() : nullptr,
      debug_buffer_size,
      program_verification);
}

static constexpr size_t kDEFAULT_BUNDLED_INPUT_POOL_SIZE = 16 * 1024U;

struct PyBundledModule final {
  explicit PyBundledModule(
      const py::bytes& buffer,
      uint32_t bundled_input_pool_size)
      : bundled_program_ptr_(buffer),
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

 private:
  // Store the bytes object instead of a raw pointer so that this module will
  // keep the bytes alive.
  const py::bytes bundled_program_ptr_;
  const void* program_ptr_;
  size_t program_len_;
};

/// Expose a subset of TensorInfo information to python.
struct PyTensorInfo final {
  explicit PyTensorInfo(
      std::shared_ptr<Module> module,
      torch::executor::TensorInfo info)
      : module_(std::move(module)), info_(info) {}

  py::tuple sizes() const {
    const auto shape = info_.sizes();
    py::tuple tup(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      tup[i] = py::cast(shape[i]);
    }
    return tup;
  }

  int8_t dtype() const {
    return static_cast<std::underlying_type<exec_aten::ScalarType>::type>(
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
  // TensorInfo relies on module to be alive.
  std::shared_ptr<Module> module_;
  torch::executor::TensorInfo info_;
};

/// Expose a subset of MethodMeta information to python.
struct PyMethodMeta final {
  explicit PyMethodMeta(
      std::shared_ptr<Module> module,
      torch::executor::MethodMeta meta)
      : module_(std::move(module)), meta_(meta) {}

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
    return std::make_unique<PyTensorInfo>(module_, result.get());
  }

  size_t num_outputs() const {
    return meta_.num_outputs();
  }

  std::unique_ptr<PyTensorInfo> output_tensor_meta(size_t index) const {
    const auto result = meta_.output_tensor_meta(index);
    THROW_INDEX_IF_ERROR(
        result.error(), "Cannot get output tensor meta at %zu", index);
    return std::make_unique<PyTensorInfo>(module_, result.get());
  }

  py::str repr() const {
    py::list input_meta_strs;
    for (size_t i = 0; i < meta_.num_inputs(); ++i) {
      input_meta_strs.append(py::str(input_tensor_meta(i)->repr()));
    }
    py::list output_meta_strs;
    for (size_t i = 0; i < meta_.num_outputs(); ++i) {
      output_meta_strs.append(py::str(output_tensor_meta(i)->repr()));
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
  // Must keep the Module object alive or else the meta object is invalidated.
  std::shared_ptr<Module> module_;
  torch::executor::MethodMeta meta_;
};

struct PyModule final {
  explicit PyModule(
      const py::bytes& buffer,
      bool enable_etdump,
      size_t debug_buffer_size = 0,
      Program::Verification program_verification =
          Program::Verification::InternalConsistency)
      : module_(load_module_from_buffer(
            buffer.cast<std::string_view>().data(),
            py::len(buffer),
            enable_etdump,
            debug_buffer_size,
            program_verification)) {}

  explicit PyModule(
      const void* ptr,
      size_t ptr_len,
      bool enable_etdump,
      size_t debug_buffer_size = 0,
      Program::Verification program_verification =
          Program::Verification::InternalConsistency)
      : module_(load_module_from_buffer(
            ptr,
            ptr_len,
            enable_etdump,
            debug_buffer_size,
            program_verification)) {}

  explicit PyModule(
      const std::string& path,
      bool enable_etdump,
      size_t debug_buffer_size = 0,
      Program::Verification program_verification =
          Program::Verification::InternalConsistency)
      : module_(load_module_from_file(
            path,
            enable_etdump,
            debug_buffer_size,
            program_verification)) {}

  PyModule(const PyModule&) = delete;
  PyModule& operator=(const PyModule&) = delete;
  PyModule(PyModule&&) = default;
  PyModule& operator=(PyModule&&) = default;

  // Module is only valid as long as the python buffer is alive.
  static std::unique_ptr<PyModule> load_from_buffer(
      const py::bytes& buffer,
      bool enable_etdump,
      size_t debug_buffer_size = 0,
      Program::Verification program_verification =
          Program::Verification::InternalConsistency) {
    return std::make_unique<PyModule>(
        buffer, enable_etdump, debug_buffer_size, program_verification);
  }
  static std::unique_ptr<PyModule> load_from_file(
      const std::string& path,
      bool enable_etdump,
      size_t debug_buffer_size = 0,
      Program::Verification program_verification =
          Program::Verification::InternalConsistency) {
    return std::make_unique<PyModule>(
        path, enable_etdump, debug_buffer_size, program_verification);
  }

  static std::unique_ptr<PyModule> load_from_bundled_program(
      PyBundledModule& m,
      bool enable_etdump,
      size_t debug_buffer_size = 0) {
    return std::make_unique<PyModule>(
        m.get_program_ptr(),
        m.get_program_len(),
        enable_etdump,
        debug_buffer_size);
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
        // alias_etensor_to_attensor will assert on this later, so to better
        // propogate up to python we check early and throw an exception.
        if (!at_tensor.is_contiguous()) {
          auto error_msg = "Input " + std::to_string(i) + "for method " +
              method_name + " is not contiguous.";
          throw std::runtime_error(error_msg);
        }

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

        // Only works for MemoryFormat::Contiguous inputs
        std::vector<torch::executor::Tensor::DimOrderType> dim_order;
        for (size_t cur_dim = 0; cur_dim < dim; cur_dim++) {
          dim_order.push_back(cur_dim);
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
        // Unsupported pytype
        ET_ASSERT_UNREACHABLE_MSG(type_str.c_str());
      }
    }

    const auto& method = module_->get_method(method_name);
    const auto num_outputs = method.outputs_size();
    output_storages_ = make_output_storages(method);
    std::vector<Span<uint8_t>> output_storage_spans(num_outputs);
    for (int i = 0; i < output_storages_.size(); ++i) {
      output_storage_spans[i] =
          Span<uint8_t>(output_storages_[i].data(), output_storages_[i].size());
    }
    auto outputs =
        module_->run_method(method_name, cpp_inputs, output_storage_spans);

    // Retrieve outputs
    return get_outputs_as_py_list(outputs, clone_outputs);
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
    return module_->has_etdump();
  }

  void write_etdump_result_to_file(
      const std::string& path,
      const py::object& debug_buffer_path) {
    if (!has_etdump()) {
      throw std::runtime_error("No etdump found");
    }
    auto& etdump = module_->etdump();
    etdump_result result = etdump.get_etdump_data();
    if (result.buf != nullptr && result.size > 0) {
      write_data_to_file(path, result.buf, result.size);
      free(result.buf);
      if (module_->has_etdump_debug_buffer() &&
          py::isinstance<py::str>(debug_buffer_path)) {
        // Also write out the debug buffer to a separate file if requested.
        std::string debug_buffer_path_str =
            py::cast<py::str>(debug_buffer_path);
        const auto debug_buffer = module_->get_etdump_debug_buffer();
        write_data_to_file(
            debug_buffer_path_str, debug_buffer.data(), debug_buffer.size());
      }
    } else {
      ET_LOG(
          Info,
          "No etdump data found, try rebuilding with "
          "the CMake option EXECUTORCH_ENABLE_EVENT_TRACER or with "
          "buck run --config executorch.event_tracer_enabled=true");
    }
  }

  void load_bundled_input(
      PyBundledModule& m,
      const std::string method_name,
      size_t testset_idx) {
    const void* bundled_program_ptr = m.get_bundled_program_ptr();
    Error status = executorch::bundled_program::load_bundled_input(
        module_->get_method(method_name), bundled_program_ptr, testset_idx);
    THROW_IF_ERROR(
        status,
        "load_bundled_input failed with status 0x%" PRIx32,
        static_cast<uint32_t>(status));
  }

  py::list verify_result_with_bundled_expected_output(
      PyBundledModule& m,
      const std::string method_name,
      size_t testset_idx,
      double rtol = 1e-5,
      double atol = 1e-8) {
    const void* bundled_program_ptr = m.get_bundled_program_ptr();
    auto& method = module_->get_method(method_name);
    Error status = executorch::bundled_program::load_bundled_input(
        method, bundled_program_ptr, testset_idx);
    THROW_IF_ERROR(
        status,
        "load_bundled_input failed with status 0x%" PRIx32,
        static_cast<uint32_t>(status));
    py::list outputs = plan_execute(method_name);
    status = executorch::bundled_program::verify_method_outputs(
        method, bundled_program_ptr, testset_idx, rtol, atol);
    THROW_IF_ERROR(
        status,
        "Result verification failed with status %" PRIu32,
        static_cast<uint32_t>(status));
    return outputs;
  }

  py::list plan_execute(
      const std::string method_name,
      bool clone_outputs = true) {
    auto& method = module_->get_method(method_name);
    // Need to pre-allocate space for outputs just like in run_method.
    const auto num_outputs = method.outputs_size();
    output_storages_ = make_output_storages(method);
    std::vector<Span<uint8_t>> output_storage_spans(num_outputs);
    for (int i = 0; i < output_storages_.size(); ++i) {
      output_storage_spans[i] =
          Span<uint8_t>(output_storages_[i].data(), output_storages_[i].size());
    }
    setup_output_storage(method, output_storage_spans);
    auto status = method.execute();
    THROW_IF_ERROR(
        status,
        "executing execution plan for method 'forward' failed with error: 0x%" PRIx32,
        static_cast<uint32_t>(status));
    const auto outputs = module_->get_outputs(method_name);
    return get_outputs_as_py_list(outputs, clone_outputs);
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

  std::unique_ptr<PyMethodMeta> method_meta(const std::string method_name) {
    auto& method = module_->get_method(method_name);
    return std::make_unique<PyMethodMeta>(module_, method.method_meta());
  }

  std::vector<std::string> method_names() {
    return module_->method_names();
  }

 private:
  std::shared_ptr<Module> module_;
  // Need to keep-alive output storages until they can be compared in case of
  // bundled programs.
  std::vector<std::vector<uint8_t>> output_storages_;

  std::vector<std::vector<uint8_t>> make_output_storages(const Method& method) {
    const auto num_outputs = method.outputs_size();
    // These output storages will not be used if the ExecuTorch program already
    // pre-allocated output space. That is represented by an error from
    // set_output_data_ptr.
    std::vector<std::vector<uint8_t>> output_storages(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
      const auto& output_tensor_meta =
          method.method_meta().output_tensor_meta(i);
      if (!output_tensor_meta.ok()) {
        // If the output isn't a tensor it won't have a tensor meta.
        ET_LOG(
            Error,
            "Tensor meta doesn't exist for output %zu, error is 0x%" PRIx32
            ", skipping allocating storage",
            i,
            static_cast<uint32_t>(output_tensor_meta.error()));
        output_storages[i] = std::vector<uint8_t>();
        continue;
      }
      const size_t output_size = output_tensor_meta.get().nbytes();
      output_storages[i] = std::vector<uint8_t>(output_size);
    }
    return output_storages;
  }
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
      py::arg("path"),
      py::arg("enable_etdump") = false,
      py::arg("debug_buffer_size") = 0,
      py::arg("program_verification") =
          Program::Verification::InternalConsistency,
      call_guard);
  m.def(
      "_load_for_executorch_from_buffer",
      &PyModule::load_from_buffer,
      py::arg("buffer"),
      py::arg("enable_etdump") = false,
      py::arg("debug_buffer_size") = 0,
      py::arg("program_verification") =
          Program::Verification::InternalConsistency,
      call_guard);
  m.def(
      "_load_for_executorch_from_bundled_program",
      &PyModule::load_from_bundled_program,
      py::arg("ptr"),
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
  m.def("_get_operator_names", &get_operator_names);
  m.def("_create_profile_block", &create_profile_block, call_guard);
  m.def(
      "_reset_profile_results",
      []() { EXECUTORCH_RESET_PROFILE_RESULTS(); },
      call_guard);

  py::class_<PyModule>(m, "ExecuTorchModule")
      .def("load_bundled_input", &PyModule::load_bundled_input, call_guard)
      .def(
          "verify_result_with_bundled_expected_output",
          &PyModule::verify_result_with_bundled_expected_output,
          py::arg("bundle"),
          py::arg("method_name"),
          py::arg("testset_idx"),
          py::arg("rtol") = 1e-5,
          py::arg("atol") = 1e-8,
          call_guard)
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

  py::class_<PyBundledModule>(m, "BundledModule");
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
      .def("__repr__", &PyMethodMeta::repr, call_guard);
}

} // namespace pybindings
} // namespace extension
} // namespace executorch
