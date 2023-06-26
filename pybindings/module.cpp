#include <cstdio>
#include <stdexcept>
#include <unordered_map>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <executorch/core/Assert.h>
#include <executorch/core/Constants.h>
#include <executorch/core/DataLoader.h>
#include <executorch/core/OperatorRegistry.h>
#include <executorch/core/Runtime.h>
#include <executorch/executor/Executor.h>
#include <executorch/executor/Program.h>
#include <executorch/profiler/profiler.h>
#include <executorch/schema/bundled_program_schema_generated.h>
#include <executorch/schema/schema_generated.h>
#include <executorch/util/TestMemoryConfig.h>
#include <executorch/util/bundled_program_verification.h>
#include <executorch/util/embedded_data_loader.h>
#include <executorch/util/mmap_data_loader.h>
#include <executorch/util/read_file.h>

#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/core/functional.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>

#ifndef USE_ATEN_LIB
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <executorch/util/aten_bridge.h>
#endif

/// Throws a runtime_error with the provided message if `error` is not `Ok`.
#define THROW_IF_ERROR(error, message, ...)                       \
  ({                                                              \
    if ((error) != torch::executor::Error::Ok) {                  \
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

using util::EmbeddedDataLoader;
using util::MmapDataLoader;

class Module final {
  using run_method_inputs_type = const std::vector<EValue>&;
  using run_method_return_type = std::vector<EValue>;

 public:
  Module(std::unique_ptr<DataLoader> loader, MemoryManager* memory_manager)
      : loader_(std::move(loader)) {
    torch::executor::runtime_init();
    Result<Program> program = Program::Load(
        loader_.get(), Program::Verification::InternalConsistency);
    THROW_IF_ERROR(
        program.error(),
        "Failed to deserialize program: 0x%" PRIx32,
        program.error());
    program_ = std::make_unique<Program>(std::move(program.get()));
    for (size_t i = 0; i < program_->num_methods(); ++i) {
      auto name = program_->get_method_name(i);
      // It's safe to use the same memory manager for all modules because
      // we can guarantee that only one will be executing at a time.
      // Everything in this module runs on a single thread.
      auto executor =
          std::make_unique<Executor>(program_.get(), memory_manager);
      auto status = executor->init_execution_plan(name.get());
      THROW_IF_ERROR(
          status,
          "initializing executor for method %s failed with error 0x:%" PRIx32,
          name.get(),
          status);
      methods_.insert({std::string(name.get()), std::move(executor)});
    }
  }

  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;
  Module(Module&&) = default;
  Module& operator=(Module&&) = default;

  template <typename... Types>
  run_method_return_type run_method(
      const std::string& method_name,
      Types&&... args) {
    return run_method_internal(method_name, std::vector<EValue>{args...});
  }

  run_method_return_type forward(run_method_inputs_type args) {
    return run_method("forward", args);
  }

  template <typename... Types>
  run_method_return_type forward(Types&&... args) {
    return run_method("forward", std::forward<Types>(args)...);
  }

  void set_delete_memory(std::shared_ptr<char> mem_to_delete) {
    mem_to_delete_ = mem_to_delete;
  }

  // TODO(T148052221): Only used to support the bundled input functions. Remove
  // this method
  ExecutionPlan& get_forward_execution_plan() {
    return methods_["forward"]->execution_plan();
  }

  // TODO(T148052221): Only used to support the bundled input functions. Remove
  // this method
  void plan_execute() {
    auto status = methods_["forward"]->execution_plan().execute();
    THROW_IF_ERROR(
        status,
        "executing execution plan for method 'forward' failed with error: 0x%" PRIx32,
        status);
  }

 private:
  run_method_return_type run_method_internal(
      const std::string& method_name,
      run_method_inputs_type args) {
    auto& plan = methods_[method_name]->execution_plan();
    exec_aten::ArrayRef<EValue> input_evalue_list(args.data(), args.size());

    Error set_inputs_status = plan.set_inputs(input_evalue_list);
    THROW_IF_ERROR(
        set_inputs_status,
        "plan.set_inputs() for method '%s' failed with error 0x%" PRIx32,
        method_name.c_str(),
        set_inputs_status);

#ifdef USE_ATEN_LIB
    // [TLS handling] This is to workaround an assertion failure
    // (https://fburl.com/code/302jyn8d) running `gelu` in ATen mode in fbcode
    // (such as bento). The problem is Executorch ATen mode doesn't have Thread
    // Local State, but `torch-cpp` is assuming tls init is done. There are two
    // more checks: MKLDNN disabled and C10_MOBILE, if any of them is true we
    // won't be hitting this assertion error. However in `torch-cpp` lib both
    // checks are false. Production impact: this should not make any impact in
    // production environment, given that in xplat we are depending on a library
    // that enables C10_MOBILE (`torch_mobile_core`).
    c10::impl::ExcludeDispatchKeyGuard no_autograd(
        c10::autograd_dispatch_keyset);
#endif
    Error execute_status = plan.execute();
    THROW_IF_ERROR(
        execute_status,
        "execution_plan().execute() failed with error 0x%" PRIx32,
        execute_status);
    // process outputs
    std::vector<EValue> result(plan.outputs_size());

    Error get_outputs_status =
        plan.get_outputs(result.data(), plan.outputs_size());
    THROW_IF_ERROR(
        get_outputs_status,
        "plan.get_outputs() for method '%s' failed with error 0x%" PRIx32,
        method_name.c_str(),
        get_outputs_status);

    return result;
  }

  std::shared_ptr<char> mem_to_delete_; // loader_ may point to this.
  std::shared_ptr<DataLoader> loader_; // program_ points to this.
  std::unique_ptr<const Program> program_; // executor_ points to this.
  std::unordered_map<std::string, std::unique_ptr<Executor>> methods_;
};

inline std::unique_ptr<Module> load_from_buffer(
    std::shared_ptr<char> ptr,
    size_t ptr_len,
    MemoryManager* memory_manager) {
  EXECUTORCH_SCOPE_PROF("load_from_buffer");
  auto loader = std::make_unique<EmbeddedDataLoader>(ptr.get(), ptr_len);
  auto m = std::make_unique<Module>(std::move(loader), memory_manager);
  m->set_delete_memory(std::move(ptr));
  return m;
}

inline std::unique_ptr<Module> load_from_buffer(
    const void* ptr,
    size_t ptr_len,
    MemoryManager* memory_manager) {
  EXECUTORCH_SCOPE_PROF("load_from_buffer");
  auto loader = std::make_unique<EmbeddedDataLoader>(ptr, ptr_len);
  return std::make_unique<Module>(std::move(loader), memory_manager);
}

inline std::unique_ptr<Module> load_from_file(
    const std::string& path,
    MemoryManager* memory_manager) {
  EXECUTORCH_SCOPE_PROF("load_from_file");

  Result<MmapDataLoader> res = MmapDataLoader::From(path.c_str());
  THROW_IF_ERROR(
      res.error(),
      "Failed to create MmapDataLoader from file %s, error: 0x:%" PRIx32,
      path.c_str(),
      res.error());

  auto loader = std::make_unique<MmapDataLoader>(std::move(res.get()));
  return std::make_unique<Module>(std::move(loader), memory_manager);
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
    return py::cast(v.toTensor());
#else
    return py::cast(
        torch::util::atTensorFromETensor(
            v.toTensor().unsafeGetTensorImpl(), keep_alive.sizes),
        py::return_value_policy::reference);
#endif
  }
  ET_ASSERT_UNREACHABLE();
}

static constexpr size_t kDEFAULT_NON_CONSTANT_POOL_SIZE = 256 * kMB;
static constexpr size_t kRUNTIME_POOL_SIZE = 256 * kMB;
static constexpr size_t kDEFAULT_BUNDLED_INPUT_POOL_SIZE = 16 * kKB;

struct PyBundledModule final {
  explicit PyBundledModule(
      const py::bytes& buffer,
      uint32_t bundled_input_pool_size)
      : bundled_program_ptr_(
            static_cast<const void*>((buffer.cast<std::string_view>().data()))),
        program_ptr_(static_cast<const void*>(
            executorch::GetBundledProgram(bundled_program_ptr_)
                ->program()
                ->data())),
        program_len_(executorch::GetBundledProgram(bundled_program_ptr_)
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
  explicit PyModule(
      const py::bytes& buffer,
      const py::int_ non_const_pool_size,
      const py::int_ runtime_pool_size)
      : memory_manager_creator_(non_const_pool_size, runtime_pool_size),
        memory_manager_(memory_manager_creator_.get_memory_manager()),
        module_(torch::executor::load_from_buffer(
            buffer.cast<std::string_view>().data(),
            py::len(buffer),
            memory_manager_)) {}

  explicit PyModule(
      const void* ptr,
      size_t ptr_len,
      const py::int_ non_const_pool_size,
      const py::int_ runtime_pool_size)
      : memory_manager_creator_(non_const_pool_size, runtime_pool_size),
        memory_manager_(memory_manager_creator_.get_memory_manager()),
        module_(
            torch::executor::load_from_buffer(ptr, ptr_len, memory_manager_)) {}

  explicit PyModule(
      const std::string& path,
      const py::int_ non_const_pool_size,
      const py::int_ runtime_pool_size)
      : memory_manager_creator_(non_const_pool_size, runtime_pool_size),
        memory_manager_(memory_manager_creator_.get_memory_manager()),
        module_(torch::executor::load_from_file(path, memory_manager_)) {}

  PyModule(const PyModule&) = delete;
  PyModule& operator=(const PyModule&) = delete;
  PyModule(PyModule&&) = default;
  PyModule& operator=(PyModule&&) = default;

  // Module is valid until buffer element is not destructed, take ownership/copy
  // of the buffer?
  static std::unique_ptr<PyModule> load_from_buffer(
      const py::bytes& buffer,
      const py::int_ non_const_pool_size = kDEFAULT_NON_CONSTANT_POOL_SIZE,
      const py::int_ runtime_pool_size = kRUNTIME_POOL_SIZE) {
    return std::make_unique<PyModule>(
        buffer, non_const_pool_size, runtime_pool_size);
  }
  static std::unique_ptr<PyModule> load_from_file(
      const std::string& path,
      const py::int_ non_const_pool_size = kDEFAULT_NON_CONSTANT_POOL_SIZE,
      const py::int_ runtime_pool_size = kRUNTIME_POOL_SIZE) {
    return std::make_unique<PyModule>(
        path, non_const_pool_size, runtime_pool_size);
  }

  static std::unique_ptr<PyModule> load_from_bundled_program(
      PyBundledModule& m,
      const py::int_ non_const_pool_size = kDEFAULT_NON_CONSTANT_POOL_SIZE,
      const py::int_ runtime_pool_size = kRUNTIME_POOL_SIZE) {
    return std::make_unique<PyModule>(
        m.get_program_ptr(),
        m.get_program_len(),
        non_const_pool_size,
        runtime_pool_size);
  }

  void
  load_bundled_input(PyBundledModule& m, size_t plan_idx, size_t testset_idx) {
    const void* bundled_program_ptr = m.get_bundled_program_ptr();
    Error status = util::LoadBundledInput(
        module_->get_forward_execution_plan(),
        bundled_program_ptr,
        &m.get_bundled_input_allocator(),
        plan_idx,
        testset_idx);
    ET_CHECK_MSG(
        status == Error::Ok,
        "LoadBundledInput failed with status %" PRIu32,
        status);
  }

  void verify_result_with_bundled_expected_output(
      PyBundledModule& m,
      size_t plan_idx,
      size_t testset_idx) {
    const void* bundled_program_ptr = m.get_bundled_program_ptr();
    Error status = util::VerifyResultWithBundledExpectedOutput(
        module_->get_forward_execution_plan(),
        bundled_program_ptr,
        &m.get_bundled_input_allocator(),
        plan_idx,
        testset_idx);
    ET_CHECK_MSG(
        status == Error::Ok,
        "Result verification failed with status %" PRIu32,
        status);
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

  // TODO(T148052221): Only used to support the bundled input functions. Remove
  // this method
  void plan_execute() {
    module_->plan_execute();
  }

  py::list forward(const py::sequence& pyinputs) {
    return run_method("forward", pyinputs);
  }

  const Module& module() {
    return *module_;
  }

 private:
  MemoryManagerCreatorDynamic memory_manager_creator_;
  MemoryManager* memory_manager_;
  std::unique_ptr<Module> module_;
  KeepAlive keep_alive_;
};

// TODO(T148052221): Remove this method
void load_bundled_input(
    ExecutionPlan& plan,
    PyBundledModule& m,
    size_t plan_idx,
    size_t testset_idx) {
  throw std::runtime_error(
      "This method to load bundled will be deleted by end of H1 2023.\n"
      "Please instead use module bound method, e.g.\n"
      "m = _load_for_executorch_from_buffer\n"
      "m.load_bundled_input(...)");
}

// TODO(T148052221): Remove this method
void verify_result_with_bundled_expected_output(
    ExecutionPlan& plan,
    PyBundledModule& m,
    size_t plan_idx,
    size_t testset_idx) {
  throw std::runtime_error(
      "This method to verify result with bundled output will be deleted by end of H1 2023.\n"
      "Please instead use module bound method, e.g.\n"
      "m = _load_for_executorch_from_buffer\n"
      "m.verify_result_with_bundled_expected_output(...)");
}

void create_profile_block(const std::string& name) {
  EXECUTORCH_PROFILE_CREATE_BLOCK(name.c_str());
}

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
  m.def(
      "_load_for_executorch",
      PyModule::load_from_file,
      py::arg("path"),
      py::arg("non_const_pool_size") = kDEFAULT_NON_CONSTANT_POOL_SIZE,
      py::arg("runtime_pool_size") = kRUNTIME_POOL_SIZE);
  m.def(
      "_load_for_executorch_from_buffer",
      &PyModule::load_from_buffer,
      py::arg("buffer"),
      py::arg("non_const_pool_size") = kDEFAULT_NON_CONSTANT_POOL_SIZE,
      py::arg("runtime_pool_size") = kRUNTIME_POOL_SIZE);
  m.def(
      "_load_for_executorch_from_bundled_program",
      &PyModule::load_from_bundled_program,
      py::arg("ptr"),
      py::arg("non_const_pool_size") = kDEFAULT_NON_CONSTANT_POOL_SIZE,
      py::arg("runtime_pool_size") = kRUNTIME_POOL_SIZE);
  m.def(
      "_load_bundled_program_from_buffer",
      &PyBundledModule::load_from_buffer,
      py::arg("buffer"),
      py::arg("non_const_pool_size") = kDEFAULT_BUNDLED_INPUT_POOL_SIZE);
  m.def("_load_bundled_input", &load_bundled_input);
  m.def(
      "_verify_result_with_bundled_expected_output",
      &verify_result_with_bundled_expected_output);
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
      .def("load_bundled_input", &PyModule::load_bundled_input)
      .def(
          "verify_result_with_bundled_expected_output",
          &PyModule::verify_result_with_bundled_expected_output)
      .def("plan_execute", &PyModule::plan_execute)
      .def("run_method", &PyModule::run_method)
      .def("forward", &PyModule::forward)
      .def_property_readonly_static("FORWARD_METHOD_INDEX", [](py::object) {
        return torch::executor::Program::kForwardMethodIndex;
      });
  ;

  py::class_<ExecutionPlan>(m, "ExecutionPlanWrapper")
      .def("inputs_size", &ExecutionPlan::inputs_size)
      .def("outputs_size", &ExecutionPlan::outputs_size)
      .def("init", &ExecutionPlan::init);

  py::class_<PyBundledModule>(m, "BundledModule");
}

} // namespace executor
} // namespace torch
