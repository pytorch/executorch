/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/backend/backend_update_context.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <cstddef>
#include <cstring>

using executorch::runtime::BackendOptionsMap;

namespace executorch {
namespace runtime {

/**
 * Retrieves backend options for a specific backend.
 *
 * @param backend_name The name of the backend to get options from
 * @param backend_options The backend option objects that will be filled with
 * the populated values from the backend
 * @return Error::Ok on success, Error::NotFound if backend is not found, or
 * other error codes on failure
 */
Error get_option(
    const char* backend_name,
    executorch::runtime::Span<executorch::runtime::BackendOption>
        backend_options) {
  auto backend_class = get_backend_class(backend_name);
  if (!backend_class) {
    return Error::NotFound;
  }
  executorch::runtime::BackendUpdateContext backend_update_context;
  executorch::runtime::Span<BackendOption> backend_options_ref(
      backend_options.data(), backend_options.size());
  auto result =
      backend_class->get_option(backend_update_context, backend_options_ref);
  if (result != Error::Ok) {
    return result;
  }
  return Error::Ok;
}

/**
 * Retrieves backend options for multiple backends using a backend options map.
 *
 * @param backend_options_map The backend option map containing backend names
 * and their associated options, which will be filled with the populated values
 * from the backend
 * @return Error::Ok on success, or the first error encountered when processing
 * the entries
 */
Error get_option(
    executorch::runtime::Span<executorch::runtime::Entry> backend_options_map) {
  Error result = Error::Ok;
  for (auto& entry : backend_options_map) {
    const char* backend_name = entry.backend_name;
    auto backend_options = entry.options;
    auto result = get_option(backend_name, backend_options);
    if (result != Error::Ok) {
      return result;
    }
  }
  return Error::Ok;
}

/**
 * Sets backend options for a specific backend.
 *
 * @param backend_name The name of the backend to set options for
 * @param backend_options The backend option list containing the options
 * to set
 * @return Error::Ok on success, Error::NotFound if backend is not found, or
 * other error codes on failure
 */
Error set_option(
    const char* backend_name,
    const executorch::runtime::Span<executorch::runtime::BackendOption>
        backend_options) {
  auto backend_class = get_backend_class(backend_name);
  if (!backend_class) {
    return Error::NotFound;
  }

  executorch::runtime::BackendUpdateContext backend_update_context;
  Error result =
      backend_class->set_option(backend_update_context, backend_options);
  if (result != Error::Ok) {
    return result;
  }
  return Error::Ok;
}

/**
 * Sets backend options for multiple backends using a backend options map.
 *
 * @param backend_options_map The backend option map containing backend names
 * and their associated backend options to set
 * @return Error::Ok on success, or the first error encountered when processing
 */
Error set_option(const executorch::runtime::Span<executorch::runtime::Entry>
                     backend_options_map) {
  Error result = Error::Ok;
  for (const auto& entry : backend_options_map) {
    const char* backend_name = entry.backend_name;
    auto backend_options = entry.options;
    result = set_option(backend_name, backend_options);

    if (result != Error::Ok) {
      return result;
    }
  }
  return Error::Ok;
}

} // namespace runtime
} // namespace executorch
