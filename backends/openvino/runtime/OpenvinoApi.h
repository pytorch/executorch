/*
 * Copyright (c) Intel Corporation
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <cstddef>
#include <cstdint>
#include <memory>

namespace executorch {
namespace backends {
namespace openvino {

// Forward declarations matching the OpenVINO C API opaque types.
// Only pointer types are used so struct layout is irrelevant.
typedef struct ov_core ov_core_t;
typedef struct ov_compiled_model ov_compiled_model_t;
typedef struct ov_infer_request ov_infer_request_t;
typedef struct ov_tensor ov_tensor_t;

// Value types reproduced from openvino/c/ov_shape.h and ov_common.h.
// These are stable C ABI — pinned via version constraint in pyproject.toml.
typedef struct {
  int64_t rank;
  int64_t* dims;
} ov_shape_t;

typedef struct {
  char** devices;
  size_t size;
} ov_available_devices_t;

// Intentionally partial — only OV_STATUS_OK is needed for success checks.
// The full enum is defined in openvino/c/ov_common.h.
typedef enum {
  OV_STATUS_OK = 0,
  OV_STATUS_GENERAL_ERROR = -1,
} ov_status_e;

// Values aligned with ov::element::Type_t (sequential enum).
typedef enum {
  OV_ELEMENT_UNDEFINED = 0,
  OV_ELEMENT_BOOLEAN = 1,
  OV_ELEMENT_BF16 = 2,
  OV_ELEMENT_F16 = 3,
  OV_ELEMENT_F32 = 4,
  OV_ELEMENT_F64 = 5,
  OV_ELEMENT_I4 = 6,
  OV_ELEMENT_I8 = 7,
  OV_ELEMENT_I16 = 8,
  OV_ELEMENT_I32 = 9,
  OV_ELEMENT_I64 = 10,
  OV_ELEMENT_U1 = 11,
  OV_ELEMENT_U2 = 12,
  OV_ELEMENT_U3 = 13,
  OV_ELEMENT_U4 = 14,
  OV_ELEMENT_U6 = 15,
  OV_ELEMENT_U8 = 16,
} ov_element_type_e;

// Function pointer types for each OpenVINO C API function we use.
using ov_core_create_fn = ov_status_e (*)(ov_core_t**);
using ov_core_free_fn = void (*)(ov_core_t*);
using ov_core_get_available_devices_fn =
    ov_status_e (*)(const ov_core_t*, ov_available_devices_t*);
using ov_available_devices_free_fn = void (*)(ov_available_devices_t*);
using ov_core_import_model_fn = ov_status_e (*)(
    const ov_core_t*,
    const char*,
    size_t,
    const char*,
    ov_compiled_model_t**);
using ov_compiled_model_create_infer_request_fn =
    ov_status_e (*)(const ov_compiled_model_t*, ov_infer_request_t**);
using ov_compiled_model_inputs_size_fn =
    ov_status_e (*)(const ov_compiled_model_t*, size_t*);
using ov_compiled_model_outputs_size_fn =
    ov_status_e (*)(const ov_compiled_model_t*, size_t*);
using ov_compiled_model_free_fn = void (*)(ov_compiled_model_t*);
using ov_infer_request_set_input_tensor_by_index_fn =
    ov_status_e (*)(ov_infer_request_t*, size_t, const ov_tensor_t*);
using ov_infer_request_set_output_tensor_by_index_fn =
    ov_status_e (*)(ov_infer_request_t*, size_t, const ov_tensor_t*);
using ov_infer_request_infer_fn = ov_status_e (*)(ov_infer_request_t*);
using ov_infer_request_free_fn = void (*)(ov_infer_request_t*);
using ov_tensor_create_from_host_ptr_fn =
    ov_status_e (*)(ov_element_type_e, ov_shape_t, void*, ov_tensor_t**);
using ov_tensor_free_fn = void (*)(ov_tensor_t*);
using ov_shape_create_fn =
    ov_status_e (*)(int64_t, const int64_t*, ov_shape_t*);
using ov_shape_free_fn = ov_status_e (*)(ov_shape_t*);

struct DlCloser {
  void operator()(void* handle) {
    if (handle) {
#ifdef _WIN32
      FreeLibrary(static_cast<HMODULE>(handle));
#else
      dlclose(handle);
#endif
    }
  }
};
using DlHandle = std::unique_ptr<void, DlCloser>;

struct OpenvinoFunctions {
  ov_core_create_fn core_create = nullptr;
  ov_core_free_fn core_free = nullptr;
  ov_core_get_available_devices_fn core_get_available_devices = nullptr;
  ov_available_devices_free_fn available_devices_free = nullptr;
  ov_core_import_model_fn core_import_model = nullptr;
  ov_compiled_model_create_infer_request_fn
      compiled_model_create_infer_request = nullptr;
  ov_compiled_model_inputs_size_fn compiled_model_inputs_size = nullptr;
  ov_compiled_model_outputs_size_fn compiled_model_outputs_size = nullptr;
  ov_compiled_model_free_fn compiled_model_free = nullptr;
  ov_infer_request_set_input_tensor_by_index_fn
      infer_request_set_input_tensor_by_index = nullptr;
  ov_infer_request_set_output_tensor_by_index_fn
      infer_request_set_output_tensor_by_index = nullptr;
  ov_infer_request_infer_fn infer_request_infer = nullptr;
  ov_infer_request_free_fn infer_request_free = nullptr;
  ov_tensor_create_from_host_ptr_fn tensor_create_from_host_ptr = nullptr;
  ov_tensor_free_fn tensor_free = nullptr;
  ov_shape_create_fn shape_create = nullptr;
  ov_shape_free_fn shape_free = nullptr;
};

} // namespace openvino
} // namespace backends
} // namespace executorch
