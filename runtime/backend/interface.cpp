/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/backend/interface.h>

namespace executorch {
namespace ET_RUNTIME_NAMESPACE {

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

size_t get_num_registered_backends() {
  return num_registered_backends;
}

Result<const char*> get_backend_name(size_t index) {
  if (index >= num_registered_backends) {
    return Error::InvalidArgument;
  }
  return registered_backends[index].name;
}

Error set_option(
    const char* backend_name,
    const executorch::runtime::Span<executorch::runtime::BackendOption>
        backend_options) {
  auto backend_class = get_backend_class(backend_name);
  if (!backend_class) {
    return Error::NotFound;
  }

  BackendOptionContext backend_option_context;
  Error result =
      backend_class->set_option(backend_option_context, backend_options);
  if (result != Error::Ok) {
    return result;
  }
  return Error::Ok;
}

Error get_option(
    const char* backend_name,
    executorch::runtime::Span<executorch::runtime::BackendOption>
        backend_options) {
  auto backend_class = get_backend_class(backend_name);
  if (!backend_class) {
    return Error::NotFound;
  }
  BackendOptionContext backend_option_context;
  executorch::runtime::Span<BackendOption> backend_options_ref(
      backend_options.data(), backend_options.size());
  auto result =
      backend_class->get_option(backend_option_context, backend_options_ref);
  if (result != Error::Ok) {
    return result;
  }
  return Error::Ok;
}

} // namespace ET_RUNTIME_NAMESPACE
} // namespace executorch
