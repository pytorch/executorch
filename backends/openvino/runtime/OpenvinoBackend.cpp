/*  Copyright (c) Intel Corporation
 *
 *  Licensed under the BSD License (the "License"); you may not use this file
 *  except in compliance with the License. See the license file found in the
 *  LICENSE file in the root directory of this source tree.
 */

#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <string>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include "OpenvinoBackend.h"

namespace executorch {
namespace backends {
namespace openvino {

namespace {

#ifdef _WIN32
constexpr const char* kDefaultLibName = "openvino_c.dll";
#else
constexpr const char* kDefaultLibName = "libopenvino_c.so";
#endif

template <typename FuncPtr>
FuncPtr load_symbol(void* handle, const char* name) {
#ifdef _WIN32
  void* sym =
      reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
  if (!sym) {
    ET_LOG(
        Error,
        "OpenVINO: failed to resolve symbol '%s': error %lu",
        name,
        GetLastError());
    return nullptr;
  }
#else
  dlerror(); // Clear any stale error state.
  void* sym = dlsym(handle, name);
  const char* err = dlerror();
  if (err) {
    ET_LOG(Error, "OpenVINO: failed to resolve symbol '%s': %s", name, err);
    return nullptr;
  }
#endif
  return reinterpret_cast<FuncPtr>(sym);
}

} // namespace

// Loading is attempted exactly once via std::call_once.  If the first attempt
// fails (e.g. library not on LD_LIBRARY_PATH), subsequent calls return false
// without retrying.  Users must fix their environment and restart the process.
bool OpenvinoBackend::ensure_loaded() const {
  std::call_once(load_flag_, [this]() {
    const char* lib_path = std::getenv("OPENVINO_LIB_PATH");
    const char* effective_path = lib_path ? lib_path : kDefaultLibName;

#ifdef _WIN32
    void* handle = static_cast<void*>(LoadLibrary(effective_path));
    if (!handle) {
      ET_LOG(
          Error,
          "OpenVINO runtime not found (LoadLibrary failed: error %lu). "
          "Ensure 'openvino_c.dll' is on your PATH "
          "(set OPENVINO_LIB_PATH), or install with: "
          "pip install \"openvino>=2025.1.0,<2026.0.0\"",
          GetLastError());
      return;
    }
#else
    void* handle = dlopen(effective_path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      ET_LOG(
          Error,
          "OpenVINO runtime not found (dlopen failed: %s). "
          "Ensure 'libopenvino_c.so' is on your library search path "
          "(set OPENVINO_LIB_PATH or LD_LIBRARY_PATH), or install with: "
          "pip install \"openvino>=2025.1.0,<2026.0.0\"",
          dlerror());
      return;
    }
#endif
    lib_handle_.reset(handle);

#define LOAD_SYM(field, symbol_name)                                  \
  ov_.field = load_symbol<decltype(ov_.field)>(handle, #symbol_name); \
  if (!ov_.field) {                                                   \
    ov_ = OpenvinoFunctions{};                                        \
    lib_handle_.reset(); /* handle is stale after this */             \
    return;                                                           \
  }

    LOAD_SYM(core_create, ov_core_create)
    LOAD_SYM(core_free, ov_core_free)
    LOAD_SYM(core_get_available_devices, ov_core_get_available_devices)
    LOAD_SYM(available_devices_free, ov_available_devices_free)
    LOAD_SYM(core_import_model, ov_core_import_model)
    LOAD_SYM(
        compiled_model_create_infer_request,
        ov_compiled_model_create_infer_request)
    LOAD_SYM(compiled_model_inputs_size, ov_compiled_model_inputs_size)
    LOAD_SYM(compiled_model_outputs_size, ov_compiled_model_outputs_size)
    LOAD_SYM(compiled_model_free, ov_compiled_model_free)
    LOAD_SYM(
        infer_request_set_input_tensor_by_index,
        ov_infer_request_set_input_tensor_by_index)
    LOAD_SYM(
        infer_request_set_output_tensor_by_index,
        ov_infer_request_set_output_tensor_by_index)
    LOAD_SYM(infer_request_infer, ov_infer_request_infer)
    LOAD_SYM(infer_request_free, ov_infer_request_free)
    LOAD_SYM(tensor_create_from_host_ptr, ov_tensor_create_from_host_ptr)
    LOAD_SYM(tensor_free, ov_tensor_free)
    LOAD_SYM(shape_create, ov_shape_create)
    LOAD_SYM(shape_free, ov_shape_free)

#undef LOAD_SYM

    loaded_ = true;
    ET_LOG(
        Info,
        "OpenVINO: runtime loaded successfully from '%s'",
        effective_path);
  });
  return loaded_;
}

bool OpenvinoBackend::is_available() const {
  if (!ensure_loaded()) {
    return false;
  }

  ov_core_t* core = nullptr;
  ov_status_e status = ov_.core_create(&core);
  if (status != OV_STATUS_OK || !core) {
    return false;
  }

  ov_available_devices_t devices = {};
  status = ov_.core_get_available_devices(core, &devices);
  bool available = (status == OV_STATUS_OK && devices.size > 0);

  if (devices.devices) {
    ov_.available_devices_free(&devices);
  }
  ov_.core_free(core);
  return available;
}

exr::Result<exr::DelegateHandle*> OpenvinoBackend::init(
    exr::BackendInitContext& context,
    exr::FreeableBuffer* processed,
    exr::ArrayRef<exr::CompileSpec> compile_specs) const {
  if (!ensure_loaded()) {
    return exr::Error::NotFound;
  }

  ov_core_t* core = nullptr;
  ov_status_e status = ov_.core_create(&core);
  if (status != OV_STATUS_OK || !core) {
    ET_LOG(Error, "OpenVINO: failed to create core (status=%d)", status);
    return exr::Error::Internal;
  }

  const char* data_ptr = static_cast<const char*>(processed->data());
  size_t data_size = processed->size();

  std::string device = "CPU";
  for (auto& compile_spec : compile_specs) {
    if (std::strcmp(compile_spec.key, "device") == 0) {
      const char* buf = static_cast<const char*>(compile_spec.value.buffer);
      size_t len = compile_spec.value.nbytes;
      // Strip trailing null bytes that may be included in nbytes.
      while (len > 0 && buf[len - 1] == '\0') {
        --len;
      }
      if (len > 0) {
        device.assign(buf, len);
      }
    }
  }

  ov_compiled_model_t* compiled_model = nullptr;
  status = ov_.core_import_model(
      core, data_ptr, data_size, device.c_str(), &compiled_model);
  ov_.core_free(core);

  if (status != OV_STATUS_OK || !compiled_model) {
    ET_LOG(
        Error,
        "OpenVINO: failed to import model for device '%s' (status=%d)",
        device.c_str(),
        status);
    return exr::Error::Internal;
  }

  processed->Free();

  ov_infer_request_t* infer_request = nullptr;
  status =
      ov_.compiled_model_create_infer_request(compiled_model, &infer_request);
  if (status != OV_STATUS_OK || !infer_request) {
    ET_LOG(
        Error, "OpenVINO: failed to create infer request (status=%d)", status);
    ov_.compiled_model_free(compiled_model);
    return exr::Error::Internal;
  }

  exr::MemoryAllocator* allocator = context.get_runtime_allocator();
  if (!allocator) {
    ET_LOG(Error, "OpenVINO: runtime allocator is null");
    ov_.infer_request_free(infer_request);
    ov_.compiled_model_free(compiled_model);
    return exr::Error::Internal;
  }
  ExecutionHandle* handle = allocator->allocateInstance<ExecutionHandle>();
  if (!handle) {
    ET_LOG(Error, "OpenVINO: failed to allocate ExecutionHandle");
    ov_.infer_request_free(infer_request);
    ov_.compiled_model_free(compiled_model);
    return exr::Error::MemoryAllocationFailed;
  }
  new (handle) ExecutionHandle;
  handle->compiled_model = compiled_model;
  handle->infer_request = infer_request;

  return handle;
}

exr::Error OpenvinoBackend::execute(
    exr::BackendExecutionContext& context,
    exr::DelegateHandle* input_handle,
    exr::Span<exr::EValue*> args) const {
  (void)context;
  ExecutionHandle* execution_handle =
      static_cast<ExecutionHandle*>(input_handle);

  size_t num_inputs = 0;
  size_t num_outputs = 0;
  ov_status_e status = ov_.compiled_model_inputs_size(
      execution_handle->compiled_model, &num_inputs);
  if (status != OV_STATUS_OK) {
    return exr::Error::Internal;
  }
  status = ov_.compiled_model_outputs_size(
      execution_handle->compiled_model, &num_outputs);
  if (status != OV_STATUS_OK) {
    return exr::Error::Internal;
  }

  // Bounds check must come after querying num_inputs/num_outputs from the
  // compiled model — those values are only known at runtime.  If either
  // query above fails, we return Internal before reaching this point.
  ET_CHECK_OR_RETURN_ERROR(
      num_inputs + num_outputs == args.size(),
      InvalidArgument,
      "OpenVINO: expected %zu args (inputs=%zu + outputs=%zu), got %zu",
      num_inputs + num_outputs,
      num_inputs,
      num_outputs,
      args.size());

  for (size_t i = 0; i < num_inputs; i++) {
    ov_tensor_t* tensor = nullptr;

    if (args[i]->isInt()) {
      int64_t* val = &(args[i]->payload.copyable_union.as_int);
      int64_t dims[] = {1};
      ov_shape_t shape = {};
      status = ov_.shape_create(1, dims, &shape);
      if (status != OV_STATUS_OK) {
        return exr::Error::Internal;
      }
      status =
          ov_.tensor_create_from_host_ptr(OV_ELEMENT_I64, shape, val, &tensor);
      ov_.shape_free(&shape);
      if (status != OV_STATUS_OK || !tensor) {
        return exr::Error::Internal;
      }
    } else if (args[i]->isTensor()) {
      auto input_tensor = args[i]->toTensor();
      auto result = create_ov_tensor(input_tensor);
      if (!result.ok()) {
        return result.error();
      }
      tensor = result.get();
    } else {
      ET_LOG(Error, "OpenVINO: unsupported input arg type at index %zu", i);
      return exr::Error::InvalidArgument;
    }

    status = ov_.infer_request_set_input_tensor_by_index(
        execution_handle->infer_request, i, tensor);
    // Safe to free: the OpenVINO C API wraps ov::Tensor in a shared_ptr
    // (see ov_tensor struct in openvino/src/bindings/c/src/common.h).
    // set_input_tensor dereferences the shared_ptr and passes by value to the
    // C++ InferRequest, which stores its own shared_ptr copy.  Freeing the
    // C wrapper here only decrements the refcount; the tensor stays alive
    // inside the infer_request until it is freed or overwritten.
    ov_.tensor_free(tensor);
    if (status != OV_STATUS_OK) {
      return exr::Error::Internal;
    }
  }

  for (size_t i = 0; i < num_outputs; i++) {
    ET_CHECK_OR_RETURN_ERROR(
        args[num_inputs + i]->isTensor(),
        InvalidArgument,
        "OpenVINO: expected tensor for output %zu",
        i);
    auto output_tensor = args[num_inputs + i]->toTensor();
    auto result = create_ov_tensor(output_tensor);
    if (!result.ok()) {
      return result.error();
    }
    ov_tensor_t* tensor = result.get();

    status = ov_.infer_request_set_output_tensor_by_index(
        execution_handle->infer_request, i, tensor);
    // Safe to free: see shared_ptr ownership comment on input tensor above.
    ov_.tensor_free(tensor);
    if (status != OV_STATUS_OK) {
      return exr::Error::Internal;
    }
  }

  status = ov_.infer_request_infer(execution_handle->infer_request);
  if (status != OV_STATUS_OK) {
    ET_LOG(Error, "OpenVINO: inference failed (status=%d)", status);
    return exr::Error::Internal;
  }

  return exr::Error::Ok;
}

// Lifecycle note: destroy() is only called for handles returned by a
// successful init(), which requires ensure_loaded() to have succeeded.
// The function-pointer null checks below are an extra safety net in case
// the library was torn down out of order (e.g. process exit).
void OpenvinoBackend::destroy(exr::DelegateHandle* handle) const {
  if (!handle) {
    return;
  }

  ExecutionHandle* execution_handle = static_cast<ExecutionHandle*>(handle);

  if (execution_handle->infer_request && ov_.infer_request_free) {
    ov_.infer_request_free(execution_handle->infer_request);
    execution_handle->infer_request = nullptr;
  }

  if (execution_handle->compiled_model && ov_.compiled_model_free) {
    ov_.compiled_model_free(execution_handle->compiled_model);
    execution_handle->compiled_model = nullptr;
  }
}

exr::Result<ov_tensor_t*> OpenvinoBackend::create_ov_tensor(
    const exa::Tensor& tensor) const {
  ov_element_type_e ov_type = convert_to_openvino_type(tensor.scalar_type());
  if (ov_type == OV_ELEMENT_UNDEFINED) {
    return exr::Error::NotSupported;
  }
  auto sizes = tensor.sizes();
  int64_t rank = sizes.size();
  ET_CHECK_OR_RETURN_ERROR(
      rank >= 0 && rank <= 1024,
      InvalidArgument,
      "OpenVINO: unreasonable tensor rank %" PRId64,
      rank);
  // Stack buffer for common ranks; heap-allocate via unique_ptr for larger.
  int64_t dims_buf[8];
  std::unique_ptr<int64_t[]> dims_heap;
  int64_t* dims = dims_buf;
  if (rank > 8) {
    dims_heap.reset(new int64_t[rank]);
    dims = dims_heap.get();
  }
  for (int64_t d = 0; d < rank; d++) {
    dims[d] = sizes[d];
  }
  // shape is zero-initialized; shape_free is only needed after a successful
  // shape_create (the zero state is safe to skip).
  ov_shape_t shape = {};
  ov_status_e status = ov_.shape_create(rank, dims, &shape);
  dims_heap.reset(); // Release heap dims (no-op if stack was used).
  if (status != OV_STATUS_OK) {
    return exr::Error::Internal;
  }

  ov_tensor_t* ov_tensor = nullptr;
  status = ov_.tensor_create_from_host_ptr(
      ov_type, shape, tensor.mutable_data_ptr(), &ov_tensor);
  ov_.shape_free(&shape);
  if (status != OV_STATUS_OK || !ov_tensor) {
    return exr::Error::Internal;
  }
  return ov_tensor;
}

ov_element_type_e OpenvinoBackend::convert_to_openvino_type(
    exa::ScalarType scalar_type) const {
  switch (scalar_type) {
    case exa::ScalarType::Float:
      return OV_ELEMENT_F32;
    case exa::ScalarType::Half:
      return OV_ELEMENT_F16;
    case exa::ScalarType::Int:
      return OV_ELEMENT_I32;
    case exa::ScalarType::Char:
      return OV_ELEMENT_I8;
    case exa::ScalarType::Byte:
      return OV_ELEMENT_U8;
    case exa::ScalarType::Long:
      return OV_ELEMENT_I64;
    case exa::ScalarType::Bool:
      return OV_ELEMENT_BOOLEAN;
    case exa::ScalarType::BFloat16:
      return OV_ELEMENT_BF16;
    case exa::ScalarType::Double:
      return OV_ELEMENT_F64;
    case exa::ScalarType::Short:
      return OV_ELEMENT_I16;
    default:
      ET_LOG(
          Error,
          "OpenVINO: unsupported scalar type %d",
          static_cast<int>(scalar_type));
      return OV_ELEMENT_UNDEFINED;
  }
}

} // namespace openvino
} // namespace backends
} // namespace executorch

namespace {
auto backend = executorch::backends::openvino::OpenvinoBackend();
executorch::runtime::Backend backend_id{"OpenvinoBackend", &backend};
static auto registered = executorch::runtime::register_backend(backend_id);
} // namespace
