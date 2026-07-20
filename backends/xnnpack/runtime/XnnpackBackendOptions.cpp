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
  } else if (strcmp(option.key, packed_cache_path_option_key) == 0) {
    std::array<char, runtime::kMaxOptionValueLength> arr{};
    const std::lock_guard<std::mutex> lock(path_mutex_);
    size_t len =
        std::min(packed_cache_path_.size(), runtime::kMaxOptionValueLength - 1);
    memcpy(arr.data(), packed_cache_path_.data(), len);
    option.value = arr;
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
  } else if (strcmp(option.key, packed_cache_path_option_key) == 0) {
    auto* val = std::get_if<std::array<char, runtime::kMaxOptionValueLength>>(
        &option.value);
    if (!val) {
      ET_LOG(Error, "XNNPACK packed cache path must be a string.");
      return Error::InvalidArgument;
    }
    // path_mutex_ also guards get_packed_cache_path so the read in
    // XNNPACKBackend::init never tears against a concurrent write here.
    const std::lock_guard<std::mutex> lock(path_mutex_);
    packed_cache_path_ = std::string(val->data());
    ET_LOG(
        Debug,
        "Setting XNNPACK packed cache path to %s.",
        packed_cache_path_.c_str());
  } else if (strcmp(option.key, save_weight_cache_on_disk_option_key) == 0) {
    auto* val = std::get_if<bool>(&option.value);
    if (!val) {
      ET_LOG(Error, "XNNPACK save_weight_cache_on_disk must be a bool.");
      return Error::InvalidArgument;
    }
    if (*val) {
      return save_weights_cache_locked();
    }
  }
  return Error::Ok;
}

XNNWeightsCacheManager& XnnpackBackendOptions::weights_cache_manager() {
  return weights_cache_manager_;
}

const XNNWeightsCacheManager& XnnpackBackendOptions::weights_cache_manager()
    const {
  return weights_cache_manager_;
}

runtime::Result<std::shared_ptr<delegate::XNNWeightsCache>>
XnnpackBackendOptions::get_or_create_weights_cache(
    const std::string& cache_file_path) {
  return weights_cache_manager_.get_or_create(cache_file_path);
}

Error XnnpackBackendOptions::save_weights_cache_locked() {
  return weights_cache_manager_.save_all();
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

std::string XnnpackBackendOptions::get_packed_cache_path() const {
  const std::lock_guard<std::mutex> lock(path_mutex_);
  return packed_cache_path_;
}

void XnnpackBackendOptions::set_packed_cache_path(const std::string& path) {
  const std::lock_guard<std::mutex> lock(path_mutex_);
  packed_cache_path_ = path;
}

} // namespace executorch::backends::xnnpack
