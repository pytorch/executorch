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

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/data_loader/mmap_data_loader.h>
#include <executorch/extension/memory_allocator/malloc_memory_allocator.h>
#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/sdk/bundled_program/bundled_program.h>
#include <executorch/sdk/bundled_program/schema/bundled_program_schema_generated.h>
#include <executorch/sdk/etdump/etdump_flatcc.h>
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

// Our logs work by writing to stderr. By default this is done through fprintf
// (as defined in Posix.cpp) which then does not show up in python environments.
// Here we override the pal to use std::cerr which can be properly redirected by
// scoped_estream_redirect.
void et_pal_emit_log_message(
    et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    __ET_UNUSED const char* function,
    size_t line,
    const char* message,
    __ET_UNUSED size_t length) {
  std::cerr << "[" << filename << ":" << line << "] " << message << std::endl;
}

namespace py = pybind11;
namespace torch {
namespace executor {

namespace {

using util::BufferDataLoader;
using util::MallocMemoryAllocator;
using util::MmapDataLoader;

class Module final {
 public:
  explicit Module(
      std::unique_ptr<DataLoader> loader,
      std::unique_ptr<ETDumpGen> tracer = nullptr)
      : loader_(std::move(loader)), event_tracer_(std::move(tracer)) {
    runtime_init();
    Result<Program> program = Program::load(
        loader_.get(), Program::Verification::InternalConsistency);
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
    auto& method = methods_[method_name];
    exec_aten::ArrayRef<EValue> input_evalue_list(args.data(), args.size());

    Error set_inputs_status = method->set_inputs(input_evalue_list);
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
      if (output_storages->size() != method->outputs_size()) {
        THROW_IF_ERROR(
            Error(),
            "number of output storages %zu does not match number of outputs %zu",
            output_storages->size(),
            method->outputs_size());
      }
      for (size_t i = 0; i < output_storages->size(); ++i) {
        Error output_status = method->set_output_data_ptr(
            (*output_storages)[i].data(), (*output_storages)[i].size(), i);
        if (output_status != Error::Ok) {
          // This can error if the outputs are already pre-allocated. Ignore
          // this error because it doesn't affect correctness, but log it.
          ET_LOG(
              Info,
              "Cannot set_output_data_ptr(): this likely means the outputs were MemoryPlanned inspect the error code to know for sure, but likely this is not an issue. 0x%" PRIx32,
              static_cast<uint32_t>(output_status));
        }
      }
    }
    Error execute_status = method->execute();
    THROW_IF_ERROR(
        execute_status,
        "method->execute() failed with error 0x%" PRIx32,
        static_cast<uint32_t>(execute_status));
    // process outputs
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
          Error(), "no such method in program: %s", method_name.c_str());
    }
    return *methods_[method_name].get();
  }

  bool has_etdump() {
    return static_cast<bool>(event_tracer_);
  }

  ETDumpGen& etdump() {
    return *event_tracer_;
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
};

inline std::unique_ptr<Module>
load_from_buffer(const void* ptr, size_t ptr_len, bool enable_etdump) {
  EXECUTORCH_SCOPE_PROF("load_from_buffer");
  auto loader = std::make_unique<BufferDataLoader>(ptr, ptr_len);
  return std::make_unique<Module>(
      std::move(loader),
      enable_etdump ? std::make_unique<torch::executor::ETDumpGen>() : nullptr);
}

inline std::unique_ptr<Module> load_from_file(
    const std::string& path,
    bool enable_etdump) {
  EXECUTORCH_SCOPE_PROF("load_from_file");

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
      enable_etdump ? std::make_unique<torch::executor::ETDumpGen>() : nullptr);
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

struct PyModule final {
  explicit PyModule(const py::bytes& buffer, bool enable_etdump)
      : module_(torch::executor::load_from_buffer(
            buffer.cast<std::string_view>().data(),
            py::len(buffer),
            enable_etdump)) {}

  explicit PyModule(const void* ptr, size_t ptr_len, bool enable_etdump)
      : module_(
            torch::executor::load_from_buffer(ptr, ptr_len, enable_etdump)) {}

  explicit PyModule(const std::string& path, bool enable_etdump)
      : module_(torch::executor::load_from_file(path, enable_etdump)) {}

  PyModule(const PyModule&) = delete;
  PyModule& operator=(const PyModule&) = delete;
  PyModule(PyModule&&) = default;
  PyModule& operator=(PyModule&&) = default;

  // Module is only valid as long as the python buffer is alive.
  static std::unique_ptr<PyModule> load_from_buffer(
      const py::bytes& buffer,
      bool enable_etdump) {
    return std::make_unique<PyModule>(buffer, enable_etdump);
  }
  static std::unique_ptr<PyModule> load_from_file(
      const std::string& path,
      bool enable_etdump) {
    return std::make_unique<PyModule>(path, enable_etdump);
  }

  static std::unique_ptr<PyModule> load_from_bundled_program(
      PyBundledModule& m,
      bool enable_etdump) {
    return std::make_unique<PyModule>(
        m.get_program_ptr(), m.get_program_len(), enable_etdump);
  }

  py::list run_method(
      const std::string& method_name,
      const py::sequence& inputs) {
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
        auto type = torch::util::torchToExecuTorchScalarType(
            at_tensor.options().dtype());
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
        torch::util::alias_etensor_to_attensor(at_tensor, temp);
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
    // These output storages will not be used if the ExecuTorch program already
    // pre-allocated output space. That is represented by an error from
    // set_output_data_ptr.
    std::vector<std::unique_ptr<uint8_t[]>> output_storages(num_outputs);
    std::vector<Span<uint8_t>> output_storage_spans(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
      const auto& output_tensor_meta =
          method.method_meta().output_tensor_meta(i);
      if (!output_tensor_meta.ok()) {
        // If the output isn't a tensor it won't have a tensor meta.
        ET_LOG(
            Info,
            "Tensor meta doesn't exist for output %zu, error is 0x%" PRIx32
            ", skipping allocating storage",
            i,
            static_cast<uint32_t>(output_tensor_meta.error()));
        output_storage_spans[i] = Span<uint8_t>();
        continue;
      }
      const size_t output_size = output_tensor_meta.get().nbytes();
      std::unique_ptr<uint8_t[]> output(new uint8_t[output_size]);
      output_storage_spans[i] = Span<uint8_t>(output.get(), output_size);
      output_storages[i] = std::move(output);
    }
    auto outputs =
        module_->run_method(method_name, cpp_inputs, output_storage_spans);

    // Retrieve outputs
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
        list[i] = py::cast(v.toTensor().clone());
#else
        list[i] = py::cast(
            torch::util::alias_attensor_to_etensor(v.toTensor()).clone());
#endif
      } else {
        ET_ASSERT_UNREACHABLE_MSG("Invalid model output type");
      }
    }
    return list;
  }

  py::list forward(const py::sequence& inputs) {
    return run_method("forward", inputs);
  }

  py::list forward_single_input(const torch::Tensor& inputTensor) {
    py::list py_list;
    py_list.append(py::cast(inputTensor));
    return run_method("forward", py_list);
  }

  bool has_etdump() {
    return module_->has_etdump();
  }

  void write_etdump_result_to_file(const std::string& path) {
    if (!has_etdump()) {
      throw std::runtime_error("No etdump found");
    }
    etdump_result result = module_->etdump().get_etdump_data();
    if (result.buf != nullptr && result.size > 0) {
      FILE* f = fopen(path.c_str(), "w+");
      fwrite((uint8_t*)result.buf, 1, result.size, f);
      fclose(f);
      free(result.buf);
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
      const string method_name,
      size_t testset_idx) {
    const void* bundled_program_ptr = m.get_bundled_program_ptr();
    Error status = bundled_program::LoadBundledInput(
        module_->get_method(method_name), bundled_program_ptr, testset_idx);
    THROW_IF_ERROR(
        status,
        "LoadBundledInput failed with status %" PRIu32,
        static_cast<uint32_t>(status));
  }

  void verify_result_with_bundled_expected_output(
      PyBundledModule& m,
      const string method_name,
      size_t testset_idx,
      double rtol = 1e-5,
      double atol = 1e-8) {
    const void* bundled_program_ptr = m.get_bundled_program_ptr();
    Error status = bundled_program::VerifyResultWithBundledExpectedOutput(
        module_->get_method(method_name),
        bundled_program_ptr,
        testset_idx,
        rtol,
        atol);
    THROW_IF_ERROR(
        status,
        "Result verification failed with status %" PRIu32,
        static_cast<uint32_t>(status));
  }

  void plan_execute(const string method_name) {
    auto status = module_->get_method(method_name).execute();
    THROW_IF_ERROR(
        status,
        "executing execution plan for method 'forward' failed with error: 0x%" PRIx32,
        static_cast<uint32_t>(status));
  }

 private:
  std::unique_ptr<Module> module_;
};

void create_profile_block(const std::string& name) {
  EXECUTORCH_PROFILE_CREATE_BLOCK(name.c_str());
}

py::list get_operator_names() {
  ArrayRef<Kernel> kernels = get_kernels();
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
  m.def(
      "_load_for_executorch",
      PyModule::load_from_file,
      py::arg("path"),
      py::arg("enable_etdump") = false,
      call_guard);
  m.def(
      "_load_for_executorch_from_buffer",
      &PyModule::load_from_buffer,
      py::arg("buffer"),
      py::arg("enable_etdump") = false,
      call_guard);
  m.def(
      "_load_for_executorch_from_bundled_program",
      &PyModule::load_from_bundled_program,
      py::arg("ptr"),
      py::arg("enable_etdump") = false,
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
      .def("plan_execute", &PyModule::plan_execute, call_guard)
      .def(
          "run_method",
          &PyModule::run_method,
          py::arg("method_name"),
          py::arg("inputs") = py::list(),
          call_guard)
      .def("forward", &PyModule::forward, call_guard)
      .def("has_etdump", &PyModule::has_etdump, call_guard)
      .def(
          "write_etdump_result_to_file",
          &PyModule::write_etdump_result_to_file,
          call_guard)
      .def("__call__", &PyModule::forward, call_guard)
      .def("__call__", &PyModule::forward_single_input, call_guard);

  py::class_<PyBundledModule>(m, "BundledModule");
}

} // namespace executor
} // namespace torch
