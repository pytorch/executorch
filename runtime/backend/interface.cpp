/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/assert.h>

#ifdef _WIN32
#include <memory>
#include <windows.h>
#include <tchar.h>
#define getpid GetCurrentProcessId
#else
#include <unistd.h>
#endif

// Task t128866626: Remove global static variables.
// We want to be able to run multiple Executor instances
// and having a global registration isn't a viable solution
// in the long term.
#ifdef _WIN32

#define SHARED_MEMORY_NAME "torch_executor_backend_registry"
static std::shared_ptr<torch::executor::BackendRegistry> backend_reg;

torch::executor::BackendRegistry& getBackendRegistry() {
  if (backend_reg != nullptr) {
    return *backend_reg;
  }

  HANDLE hMapFile = OpenFileMapping(
    FILE_MAP_ALL_ACCESS,   // read/write access
    FALSE,                 // do not inherit the name
    _T(SHARED_MEMORY_NAME)  // name of mapping object
  );

  if (hMapFile == NULL) {
    // Create a new file mapping object
    hMapFile = CreateFileMapping(
      INVALID_HANDLE_VALUE,    // use paging file
      NULL,                    // default security
      PAGE_READWRITE,          // read/write access
      0,                       // maximum object size (high-order DWORD)
      sizeof(torch::executor::BackendRegistry),                // maximum object size (low-order DWORD)
      _T(SHARED_MEMORY_NAME)   // name of mapping object
    );
    if (hMapFile == NULL) {
      return *backend_reg;
    }
  }

  torch::executor::BackendRegistry* registry = (torch::executor::BackendRegistry*) MapViewOfFile(
    hMapFile,   // handle to map object
    FILE_MAP_ALL_ACCESS, // read/write permission
    0,
    0,
    sizeof(torch::executor::BackendRegistry)
  );

  if (registry == NULL) {
    return *backend_reg;
  }

  if (backend_reg == nullptr) {
    backend_reg = std::shared_ptr<torch::executor::BackendRegistry>(registry, [](torch::executor::BackendRegistry* ptr) {
      UnmapViewOfFile(ptr);
    });
  }

  return *backend_reg;
}

#else

torch::executor::BackendRegistry& getBackendRegistry();
torch::executor::BackendRegistry& getBackendRegistry() {
  static torch::executor::BackendRegistry backend_reg;
  return backend_reg;
}

#endif

namespace torch {
namespace executor {

PyTorchBackendInterface::~PyTorchBackendInterface() {}

PyTorchBackendInterface* get_backend_class(const char* name) {
  return getBackendRegistry().get_backend_class(name);
}

PyTorchBackendInterface* BackendRegistry::get_backend_class(const char* name) {
  for (size_t idx = 0; idx < registrationTableSize_; idx++) {
    Backend backend = backend_table_[idx];
    if (strcmp(backend.name_, name) == 0) {
      return backend.interface_ptr_;
    }
  }
  return nullptr;
}

Error register_backend(const Backend& backend) {
  return getBackendRegistry().register_backend(backend);
}

Error BackendRegistry::register_backend(const Backend& backend) {
  if (registrationTableSize_ >= kRegistrationTableMaxSize) {
    return Error::Internal;
  }

  // Check if the name already exists in the table
  if (this->get_backend_class(backend.name_) != nullptr) {
    return Error::InvalidArgument;
  }

  backend_table_[registrationTableSize_++] = backend;
  return Error::Ok;
}

} // namespace executor
} // namespace torch
