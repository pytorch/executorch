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

Error get_option(
    executorch::runtime::Span<executorch::runtime::Entry> backend_options_map) {
  for (auto& entry : backend_options_map) {
    const char* backend_name = entry.backend_name;
    auto backend_options = entry.options;

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
  }
  return Error::Ok;
}

Error set_option(
    const executorch::runtime::Span<executorch::runtime::Entry> backend_options_map) {
  for (const auto& entry : backend_options_map) {
    const char* backend_name = entry.backend_name;
    auto backend_options = entry.options;

    auto backend_class = get_backend_class(backend_name);
    if (!backend_class) {
      return Error::NotFound;
    }

    executorch::runtime::BackendUpdateContext backend_update_context;
    auto update_result =
        backend_class->set_option(backend_update_context, backend_options);
    if (update_result != Error::Ok) {
      return update_result;
    }
  }
  return Error::Ok;
}

} // namespace runtime
} // namespace executorch
