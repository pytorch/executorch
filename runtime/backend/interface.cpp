/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/assert.h>

namespace executorch {
namespace runtime {

PyTorchBackendInterface::~PyTorchBackendInterface() {}

// TODO(T128866626): Remove global static variables.
// We want to be able to run multiple Executor instances
// and having a global registration isn't a viable solution
// in the long term.
BackendRegistry& getBackendRegistry();
BackendRegistry& getBackendRegistry() {
  static BackendRegistry backend_reg;
  return backend_reg;
}

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

} // namespace runtime
} // namespace executorch
