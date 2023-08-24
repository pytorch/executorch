/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <unordered_map>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/data_loader/mmap_data_loader.h>
#include <executorch/extension/memory_allocator/malloc_memory_allocator.h>
#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/executor/executor.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/schema/bundled_program_schema_generated.h>
#include <executorch/schema/program_generated.h>
#include <executorch/util/TestMemoryConfig.h>
#include <executorch/util/bundled_program_verification.h>
#include <executorch/util/read_file.h>

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

namespace py = pybind11;
using ATTensor = at::Tensor;
namespace torch {
namespace executor {

namespace {

using util::BufferDataLoader;
using util::MallocMemoryAllocator;
using util::MmapDataLoader;

class Module final {
 public:
  explicit Module(std::unique_ptr<DataLoader> loader)
      : loader_(std::move(loader)) {
    runtime_init();
    Result<Program> program = Program::Load(
        loader_.get(), Program::Verification::InternalConsistency);
    THROW_IF_ERROR(
        program.error(),
        "loading program failed with error: 0x%" PRIx32,
        program.error());
    program_ = std::make_unique<Program>(std::move(program.get()));

    // Figure out the size of each non_const layer we need to support every
    // method in the program. Map will be easier to use than a list because we
    // dont know how many non_const arenas there will be
    std::map<size_t, int64_t> non_const_buffer_sizes;
    for (size_t i = 0; i < program_->num_methods(); ++i) {
      auto name = program_->get_method_name(i).get();
      auto method_meta = program_->method_meta(name).get();
      // 1 on purpose because non-const are 1 indexed
      for (size_t j = 1; j < method_meta.num_non_const_buffers(); j++) {
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

    // Load methods
    for (size_t i = 0; i < program_->num_methods(); ++i) {
      auto name = program_->get_method_name(i).get();
      // It's safe to use the same memory manager for all modules because
      // we can guarantee that only one will be executing at a time.
      // Everything in this module runs on a single thread.
      Result<Method> method =
          program_->load_method(name, memory_->mem_manager());
      THROW_IF_ERROR(
          method.error(),
          "loading method %s failed with error 0x%" PRIx32,
          name,
          method.error());
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
  template <typename... Types>
  std::vector<EValue> run_method(
      const std::string& method_name,
      Types&&... args) {
    return run_method_internal(method_name, std::vector<EValue>{args...});
  }

 private:
  std::vector<EValue> run_method_internal(
      const std::string& method_name,
      const std::vector<EValue>& args) {
    auto& method = methods_[method_name];
    exec_aten::ArrayRef<EValue> input_evalue_list(args.data(), args.size());

    Error set_inputs_status = method->set_inputs(input_evalue_list);
    THROW_IF_ERROR(
        set_inputs_status,
        "method->set_inputs() for method '%s' failed with error 0x%" PRIx32,
        method_name.c_str(),
        set_inputs_status);

#ifdef USE_ATEN_LIB
    // [TLS handling] This is to workaround an assertion failure
    // (https://fburl.com/code/302jyn8d) running `gelu` in ATen mode in fbcode
    // (such as bento). The problem is Executorch ATen mode doesn't have
    // Thread Local State, but `torch-cpp` is assuming tls init is done. There
    // are two more checks: MKLDNN disabled and C10_MOBILE, if any of them is
    // true we won't be hitting this assertion error. However in `torch-cpp`
    // lib both checks are false. Production impact: this should not make any
    // impact in production environment, given that in xplat we are depending
    // on a library that enables C10_MOBILE (`torch_mobile_core`).
    c10::impl::ExcludeDispatchKeyGuard no_autograd(
        c10::autograd_dispatch_keyset);
#endif
    Error execute_status = method->execute();
    THROW_IF_ERROR(
        execute_status,
        "method->execute() failed with error 0x%" PRIx32,
        execute_status);
    // process outputs
    std::vector<EValue> result(method->outputs_size());

    Error get_outputs_status =
        method->get_outputs(result.data(), method->outputs_size());
    THROW_IF_ERROR(
        get_outputs_status,
        "method->get_outputs() for method '%s' failed with error 0x%" PRIx32,
        method_name.c_str(),
        get_outputs_status);

    return result;
  }

  /// A wrapper/util class for executorch memory allocations/manager.
  class Memory {
   public:
    explicit Memory(std::vector<std::vector<uint8_t>>&& non_const_buffers)
        : runtime_allocator_(),
          non_const_buffers_(std::move(non_const_buffers)),
          non_const_allocator_list_(create_non_const_allocators()),
          non_const_allocator_(
              non_const_allocator_list_.size(),
              non_const_allocator_list_.data()),
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

    std::vector<MemoryAllocator> non_const_allocator_list_;

    HierarchicalAllocator non_const_allocator_;

    MemoryManager mem_manager_;

    std::vector<MemoryAllocator> create_non_const_allocators() {
      std::vector<MemoryAllocator> result;
      for (size_t i = 0; i < non_const_buffers_.size(); i++) {
        result.push_back(MemoryAllocator(
            non_const_buffers_[i].size(), non_const_buffers_[i].data()));
      }
      return result;
    }
  };

  std::unique_ptr<Memory> memory_;
  std::unique_ptr<DataLoader> loader_; // program_ points to this.
  std::unique_ptr<const Program> program_; // methods_ entries points to this.
  std::unordered_map<std::string, std::unique_ptr<Method>> methods_;
};

inline std::unique_ptr<Module> load_from_buffer(
    const void* ptr,
    size_t ptr_len) {
  EXECUTORCH_SCOPE_PROF("load_from_buffer");
  auto loader = std::make_unique<BufferDataLoader>(ptr, ptr_len);
  return std::make_unique<Module>(std::move(loader));
}

inline std::unique_ptr<Module> load_from_file(const std::string& path) {
  EXECUTORCH_SCOPE_PROF("load_from_file");

  Result<MmapDataLoader> res = MmapDataLoader::From(
      path.c_str(), MmapDataLoader::MlockConfig::UseMlockIgnoreErrors);
  THROW_IF_ERROR(
      res.error(),
      "Failed to create MmapDataLoader from file %s, error: 0x:%" PRIx32,
      path.c_str(),
      res.error());

  auto loader = std::make_unique<MmapDataLoader>(std::move(res.get()));
  return std::make_unique<Module>(std::move(loader));
}

// Struct used to manage the memory of tensors allocated in lean (not aten) mode
#ifdef USE_ATEN_LIB
struct KeepAlive {};
#else
struct KeepAlive {
  std::vector<std::unique_ptr<exec_aten::TensorImpl>> tensors;
  torch::util::KeepAliveSizes sizes;
};
#endif

EValue pyToEValue(py::handle h, KeepAlive& keep_alive) {
  const std::string& type_str = py::str(h.get_type());
  EXECUTORCH_SCOPE_PROF("pyToEValue");
  if (type_str == "<class 'torch.Tensor'>") {
    auto atTensor = h.cast<ATTensor>();
#ifdef USE_ATEN_LIB
    EValue evalue(atTensor);
#else
    auto etensorImpl =
        torch::util::eTensorFromAtTensor(atTensor, keep_alive.sizes);
    EValue evalue(torch::executor::Tensor(etensorImpl.get()));
    keep_alive.tensors.push_back(std::move(etensorImpl));
#endif
    return evalue;
  } else if (py::isinstance<py::none>(h)) {
    return EValue();
  } else if (py::isinstance<py::bool_>(h)) {
    return EValue(py::cast<bool>(h));
  } else if (py::isinstance<py::int_>(h)) {
    return EValue(py::cast<int64_t>(h));
  } else {
    // Unsupported pytype
    ET_ASSERT_UNREACHABLE_MSG(type_str.c_str());
  }
}

py::object pyFromEValue(const EValue& v, KeepAlive& keep_alive) {
  EXECUTORCH_SCOPE_PROF("pyFromEValue");
  if (Tag::None == v.tag) {
    return py::none();
  } else if (Tag::Int == v.tag) {
    return py::cast(v.toInt());
  } else if (Tag::Double == v.tag) {
    return py::cast(v.toDouble());
  } else if (Tag::Bool == v.tag) {
    return py::cast(v.toBool());
  } else if (Tag::Tensor == v.tag) {
#ifdef USE_ATEN_LIB
    return py::cast(v.toTensor().clone());
#else
    // Clone so the outputs in python do not share a lifetime with the module
    // object
    return py::cast(torch::util::atTensorFromETensor(
                        v.toTensor().unsafeGetTensorImpl(), keep_alive.sizes)
                        .clone());
#endif
  }
  ET_ASSERT_UNREACHABLE();
}

static constexpr size_t kDEFAULT_BUNDLED_INPUT_POOL_SIZE = 16 * 1024U;

struct PyBundledModule final {
  explicit PyBundledModule(
      const py::bytes& buffer,
      uint32_t bundled_input_pool_size)
      : bundled_program_ptr_(
            static_cast<const void*>((buffer.cast<std::string_view>().data()))),
        program_ptr_(static_cast<const void*>(
            executorch_flatbuffer::GetBundledProgram(bundled_program_ptr_)
                ->program()
                ->data())),
        program_len_(
            executorch_flatbuffer::GetBundledProgram(bundled_program_ptr_)
                ->program()
                ->size()),
        bundled_input_allocator_(
            {bundled_input_pool_size, new uint8_t[bundled_input_pool_size]}) {}

  static std::unique_ptr<PyBundledModule> load_from_buffer(
      const py::bytes& buffer,
      uint32_t bundled_input_pool_size) {
    return std::make_unique<PyBundledModule>(buffer, bundled_input_pool_size);
  }

  MemoryAllocator& get_bundled_input_allocator() {
    return bundled_input_allocator_;
  }
  const void* get_bundled_program_ptr() {
    return bundled_program_ptr_;
  }

  const void* get_program_ptr() {
    return program_ptr_;
  }

  size_t get_program_len() {
    return program_len_;
  }

 private:
  const void* bundled_program_ptr_;
  const void* program_ptr_;
  size_t program_len_;
  MemoryAllocator bundled_input_allocator_;
};

struct PyModule final {
  explicit PyModule(const py::bytes& buffer)
      : module_(torch::executor::load_from_buffer(
            buffer.cast<std::string_view>().data(),
            py::len(buffer))) {}

  explicit PyModule(const void* ptr, size_t ptr_len)
      : module_(torch::executor::load_from_buffer(ptr, ptr_len)) {}

  explicit PyModule(const std::string& path)
      : module_(torch::executor::load_from_file(path)) {}

  PyModule(const PyModule&) = delete;
  PyModule& operator=(const PyModule&) = delete;
  PyModule(PyModule&&) = default;
  PyModule& operator=(PyModule&&) = default;

  // Module is only valid as long as the python buffer is alive.
  static std::unique_ptr<PyModule> load_from_buffer(const py::bytes& buffer) {
    return std::make_unique<PyModule>(buffer);
  }
  static std::unique_ptr<PyModule> load_from_file(const std::string& path) {
    return std::make_unique<PyModule>(path);
  }

  static std::unique_ptr<PyModule> load_from_bundled_program(
      PyBundledModule& m) {
    return std::make_unique<PyModule>(m.get_program_ptr(), m.get_program_len());
  }

  py::list run_method(const std::string& name, const py::sequence& pyinputs) {
    std::vector<EValue> inputs;
    const auto inputs_size = py::len(pyinputs);
    inputs.reserve(inputs_size);
    for (size_t i = 0; i < inputs_size; ++i) {
      inputs.emplace_back(pyToEValue(pyinputs[i], keep_alive_));
    }

    auto outputs = module_->run_method(name, inputs);

    const auto outputs_size = outputs.size();
    py::list list(outputs_size);
    for (size_t i = 0; i < outputs_size; ++i) {
      list[i] = pyFromEValue(outputs[i], keep_alive_);
    }
    return list;
  }

  py::list forward(const py::sequence& pyinputs) {
    return run_method("forward", pyinputs);
  }

 private:
  KeepAlive keep_alive_;
  std::unique_ptr<Module> module_;
};

void create_profile_block(const std::string& name) {
  EXECUTORCH_PROFILE_CREATE_BLOCK(name.c_str());
}

// Returns the list of all available ops in the Executorch runtime.
py::list get_ops_names() {
  const auto& ops_array = getOpsArray();
  py::list list(ops_array.size());
  for (size_t i = 0; i < ops_array.size(); ++i) {
    list[i] = std::string(ops_array[i].name_);
  }
  return list;
}

} // namespace

void init_module_functions(py::module_& m) {
  m.def("_load_for_executorch", PyModule::load_from_file, py::arg("path"));
  m.def(
      "_load_for_executorch_from_buffer",
      &PyModule::load_from_buffer,
      py::arg("buffer"));
  m.def(
      "_load_for_executorch_from_bundled_program",
      &PyModule::load_from_bundled_program,
      py::arg("ptr"));
  m.def(
      "_load_bundled_program_from_buffer",
      &PyBundledModule::load_from_buffer,
      py::arg("buffer"),
      py::arg("non_const_pool_size") = kDEFAULT_BUNDLED_INPUT_POOL_SIZE);
  m.def("_ops_names", &get_ops_names);
  m.def("_dump_profile_results", []() {
    prof_result_t prof_result;
    EXECUTORCH_DUMP_PROFILE_RESULTS(&prof_result);
    return py::bytes(
        reinterpret_cast<const char*>(prof_result.prof_data),
        prof_result.num_bytes);
  });
  m.def("_create_profile_block", &create_profile_block);
  m.def("_reset_profile_results", []() { EXECUTORCH_RESET_PROFILE_RESULTS(); });

  py::class_<PyModule>(m, "Module")
      .def("run_method", &PyModule::run_method)
      .def("forward", &PyModule::forward);

  py::class_<PyBundledModule>(m, "BundledModule");
}

} // namespace executor
} // namespace torch
