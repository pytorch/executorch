/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/runtime/XnnpackBackendOptions.h>

#include <cstring>

namespace executorch::backends::xnnpack {

using executorch::runtime::BackendOption;
using executorch::runtime::Error;

namespace {

// Resolve an option value, preferring the runtime spec override if present.
template <typename T>
T resolve_option(
    const ET_RUNTIME_NAMESPACE::BackendInitContext& context,
    const char* key,
    T global_default) {
  auto spec = context.get_runtime_spec<T>(key);
  if (spec.ok()) {
    return spec.get();
  }
  return global_default;
}

} // namespace

Error XnnpackBackendOptions::get_option(BackendOption& option) const {
  if (strcmp(option.key, workspace_sharing_mode_option_key) == 0) {
    option.value = static_cast<int>(sharing_mode_.load());
  } else if (strcmp(option.key, weight_cache_option_key) == 0) {
    option.value = weight_cache_enabled_.load();
  }
  return Error::Ok;
}

Error XnnpackBackendOptions::set_option(const BackendOption& option) {
  if (strcmp(option.key, workspace_sharing_mode_option_key) == 0) {
    auto* val = std::get_if<int>(&option.value);
    if (!val) {
      ET_LOG(Error, "XNNPACK workspace sharing mode must be an integer.");
      return Error::InvalidArgument;
    }
    if (*val < 0 || *val >= static_cast<int>(WorkspaceSharingMode::Count)) {
      ET_LOG(
          Error,
          "XNNPACK workspace sharing mode must be between 0 and %d, inclusive, but was %d.",
          static_cast<int>(WorkspaceSharingMode::Count) - 1,
          *val);
      return Error::InvalidArgument;
    }
    ET_LOG(Debug, "Setting XNNPACK workspace sharing mode to %d.", *val);
    sharing_mode_.store(static_cast<WorkspaceSharingMode>(*val));
  } else if (strcmp(option.key, weight_cache_option_key) == 0) {
    auto* val = std::get_if<bool>(&option.value);
    if (!val) {
      ET_LOG(Error, "XNNPACK weight cache enabled must be a bool.");
      return Error::InvalidArgument;
    }
    ET_LOG(Debug, "Setting XNNPACK weight cache enabled to %d.", *val);
    weight_cache_enabled_.store(*val);
  }
  return Error::Ok;
}

bool XnnpackBackendOptions::resolve_weight_cache(
    const ET_RUNTIME_NAMESPACE::BackendInitContext& context) const {
  return resolve_option<bool>(
      context, weight_cache_option_key, weight_cache_enabled_.load());
}

runtime::Result<WorkspaceSharingMode>
XnnpackBackendOptions::resolve_sharing_mode(
    const ET_RUNTIME_NAMESPACE::BackendInitContext& context) const {
  auto global_mode = sharing_mode_.load();
  int raw_mode = resolve_option<int>(
      context,
      workspace_sharing_mode_option_key,
      static_cast<int>(global_mode));
  if (raw_mode < 0 ||
      raw_mode >= static_cast<int>(WorkspaceSharingMode::Count)) {
    ET_LOG(
        Error,
        "XNNPACK workspace sharing mode must be between 0 and %d, inclusive, but was %d.",
        static_cast<int>(WorkspaceSharingMode::Count) - 1,
        raw_mode);
    return runtime::Error::InvalidArgument;
  }
  return static_cast<WorkspaceSharingMode>(raw_mode);
}

WorkspaceSharingMode XnnpackBackendOptions::get_sharing_mode() const {
  return sharing_mode_.load();
}

XNNWorkspaceManager& XnnpackBackendOptions::workspace_manager() {
  return workspace_manager_;
}

const XNNWorkspaceManager& XnnpackBackendOptions::workspace_manager() const {
  return workspace_manager_;
}

} // namespace executorch::backends::xnnpack
