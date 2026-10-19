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

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

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
constexpr const wchar_t* kDefaultLibName = L"openvino_c.dll";
constexpr const char* kLibrarySearchPathHint =
    "set OPENVINO_LIB_PATH to its full path, with its dependencies such as "
    "tbb12.dll beside it or in a folder added with AddDllDirectory";

// A path with a character outside the active code page has no faithful narrow
// form, so read and load through the wide APIs and convert to UTF-8 only for
// logging. Unlike _wgetenv, which reads the C runtime's copy of the
// environment, GetEnvironmentVariableW also sees a value set with
// SetEnvironmentVariableW after startup.
std::wstring configured_lib_path() {
  DWORD length = GetEnvironmentVariableW(L"OPENVINO_LIB_PATH", nullptr, 0);
  if (length == 0) {
    return std::wstring(kDefaultLibName);
  }
  std::wstring lib_path;
  // Loop, since the value may grow between asking for its size and reading it.
  while (length > lib_path.size()) {
    lib_path.resize(length);
    length =
        GetEnvironmentVariableW(L"OPENVINO_LIB_PATH", lib_path.data(), length);
  }
  lib_path.resize(length);
  return lib_path;
}

// The standard search order includes the current folder, so a bare name would
// load an openvino_c.dll planted there. The LOAD_LIBRARY_SEARCH_* flags leave
// out the current folder and PATH. For a path they also search the library's
// own folder for its dependencies, such as openvino.dll and tbb12.dll, and
// they need the path to be absolute.
void* open_library(const std::wstring& path) {
  if (path.find_first_of(L"\\/") == std::wstring::npos) {
    return LoadLibraryExW(
        path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  }
  const DWORD size = GetFullPathNameW(path.c_str(), 0, nullptr, nullptr);
  std::wstring full(size, L'\0');
  const DWORD length =
      GetFullPathNameW(path.c_str(), size, full.data(), nullptr);
  if (length == 0 || length >= size) {
    return nullptr;
  }
  full.resize(length);
  return LoadLibraryExW(
      full.c_str(),
      nullptr,
      LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
}

std::string to_utf8(const std::wstring& wide) {
  if (wide.empty()) {
    return std::string();
  }
  int size = WideCharToMultiByte(
      CP_UTF8, 0, wide.data(), (int)wide.size(), nullptr, 0, nullptr, nullptr);
  std::string out(size, '\0');
  WideCharToMultiByte(
      CP_UTF8,
      0,
      wide.data(),
      (int)wide.size(),
      out.data(),
      size,
      nullptr,
      nullptr);
  return out;
}

// The system's message for the calling thread's last error, with its code,
// since a code alone is hard to act on. Call it right after the failing call:
// later calls, logging included, may overwrite the last error.
std::string last_error() {
  const DWORD err = GetLastError();
  LPSTR buffer = nullptr;
  const DWORD flags = FORMAT_MESSAGE_ALLOCATE_BUFFER |
      FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS;
  const DWORD size = FormatMessageA(
      flags,
      nullptr,
      err,
      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      reinterpret_cast<LPSTR>(&buffer),
      0,
      nullptr);
  if (size == 0 || buffer == nullptr) {
    return "error " + std::to_string(err);
  }
  std::string message(buffer, size);
  LocalFree(buffer);
  while (!message.empty() &&
         (message.back() == '\n' || message.back() == '\r')) {
    message.pop_back();
  }
  return message + " (error " + std::to_string(err) + ")";
}

template <typename FuncPtr>
FuncPtr load_symbol(void* handle, const char* name) {
  FARPROC sym = GetProcAddress(static_cast<HMODULE>(handle), name);
  if (!sym) {
    const std::string error = last_error();
    ET_LOG(
        Error,
        "OpenVINO: failed to resolve symbol '%s': %s",
        name,
        error.c_str());
    return nullptr;
  }
  return reinterpret_cast<FuncPtr>(sym);
}
#else
constexpr const char* kDefaultLibName = "libopenvino_c.so";
constexpr const char* kLibrarySearchPathHint =
    "set OPENVINO_LIB_PATH or LD_LIBRARY_PATH";

std::string configured_lib_path() {
  const char* lib_path = std::getenv("OPENVINO_LIB_PATH");
  return lib_path ? std::string(lib_path) : std::string(kDefaultLibName);
}

void* open_library(const std::string& path) {
  return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
}

std::string to_utf8(const std::string& path) {
  return path;
}

std::string last_error() {
  const char* err = dlerror();
  return err ? err : "unknown error";
}

template <typename FuncPtr>
FuncPtr load_symbol(void* handle, const char* name) {
  dlerror(); // Clear any stale error state.
  void* sym = dlsym(handle, name);
  const char* err = dlerror();
  if (err) {
    ET_LOG(Error, "OpenVINO: failed to resolve symbol '%s': %s", name, err);
    return nullptr;
  }
  return reinterpret_cast<FuncPtr>(sym);
}
#endif

} // namespace

void DlCloser::operator()(void* handle) {
  if (!handle) {
    return;
  }
#ifdef _WIN32
  FreeLibrary(static_cast<HMODULE>(handle));
#else
  dlclose(handle);
#endif
}

OpenvinoBackend::~OpenvinoBackend() {
#ifdef _WIN32
  // The backend is a global, so in a DLL this runs inside DllMain when the DLL
  // unloads, where FreeLibrary is not safe to call. Keep the runtime loaded.
  (void)lib_handle_.release();
#endif
}

// Loading is attempted exactly once via std::call_once.  If the first attempt
// fails (e.g. library not on the search path), subsequent calls return false
// without retrying.  Users must fix their environment and restart the process.
bool OpenvinoBackend::ensure_loaded() const {
  std::call_once(load_flag_, [this]() {
    auto effective_path = configured_lib_path();

    void* handle = open_library(effective_path);
    if (!handle) {
      const std::string error = last_error();
      ET_LOG(
          Error,
          "OpenVINO runtime not found (loading '%s' failed: %s). "
          "Ensure '%s' can be found (%s), or install with: "
          "pip install \"openvino>=2025.1.0,<2027.0.0\"",
          to_utf8(effective_path).c_str(),
          error.c_str(),
          to_utf8(kDefaultLibName).c_str(),
          kLibrarySearchPathHint);
      return;
    }
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
        to_utf8(effective_path).c_str());
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
  // ov_shape_create rejects rank 0, so 0-d tensors use the zero shape as-is.
  ov_shape_t shape = {};
  ov_status_e status = OV_STATUS_OK;
  if (rank > 0) {
    status = ov_.shape_create(rank, dims, &shape);
  }
  dims_heap.reset(); // Release heap dims (no-op if stack was used).
  if (status != OV_STATUS_OK) {
    return exr::Error::Internal;
  }

  ov_tensor_t* ov_tensor = nullptr;
  status = ov_.tensor_create_from_host_ptr(
      ov_type, shape, tensor.mutable_data_ptr(), &ov_tensor);
  if (rank > 0) {
    ov_.shape_free(&shape);
  }
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
