/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/backend/interface.h>

namespace executorch {
namespace runtime {

// Pure-virtual dtors still need an implementation.
BackendInterface::~BackendInterface() {}

namespace {

// The max number of backends that can be registered globally.
constexpr size_t kMaxRegisteredBackends = 16;

// TODO(T128866626): Remove global static variables. We want to be able to run
// multiple Executor instances and having a global registration isn't a viable
// solution in the long term.

/// Global table of registered backends.
Backend registered_backends[kMaxRegisteredBackends];

/// The number of backends registered in the table.
size_t num_registered_backends = 0;

} // namespace

BackendInterface* get_backend_class(const char* name) {
  for (size_t i = 0; i < num_registered_backends; i++) {
    Backend backend = registered_backends[i];
    if (strcmp(backend.name, name) == 0) {
      return backend.backend;
    }
  }
  return nullptr;
}

Error register_backend(const Backend& backend) {
  if (num_registered_backends >= kMaxRegisteredBackends) {
    return Error::Internal;
  }

  // Check if the name already exists in the table
  if (get_backend_class(backend.name) != nullptr) {
    return Error::InvalidArgument;
  }

  registered_backends[num_registered_backends++] = backend;
  return Error::Ok;
}

} // namespace runtime
} // namespace executorch
