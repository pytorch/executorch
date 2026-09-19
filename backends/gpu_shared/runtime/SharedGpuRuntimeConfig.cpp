/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/gpu_shared/runtime/SharedGpuRuntimeConfig.h>

#include <cstring>

namespace executorch {
namespace backends {
namespace gpu_shared {
namespace {

runtime::Result<SharedContextMode> parse_context_mode(const char* value) {
  if (value == nullptr) {
    return runtime::Error::InvalidArgument;
  }
  if (std::strcmp(value, "disabled") == 0) {
    return SharedContextMode::kDisabled;
  }
  if (std::strcmp(value, "lookup_only") == 0) {
    return SharedContextMode::kLookupOnly;
  }
  if (std::strcmp(value, "lookup_or_create") == 0) {
    return SharedContextMode::kLookupOrCreate;
  }
  if (std::strcmp(value, "create_only") == 0) {
    return SharedContextMode::kCreateOnly;
  }
  return runtime::Error::InvalidArgument;
}

} // namespace

// This API is consumed by the Vulkan/VGF delegate integration follow-up PRs.
// The phase-2 runtime library intentionally has no production caller yet.
// cppcheck-suppress unusedFunction
runtime::Result<SharedGpuRuntimeConfig> parse_shared_gpu_runtime_config(
    const runtime::BackendInitContext& context) {
  SharedGpuRuntimeConfig config;

  auto token = context.get_runtime_spec<const char*>(kSharedContextTokenOption);
  if (token.ok()) {
    config.token = token.get();
  } else if (token.error() != runtime::Error::NotFound) {
    return token.error();
  }

  auto mode = context.get_runtime_spec<const char*>(kSharedContextModeOption);
  if (mode.ok()) {
    auto parsed_mode = parse_context_mode(mode.get());
    if (!parsed_mode.ok()) {
      return parsed_mode.error();
    }
    config.context_mode = parsed_mode.get();
  } else if (mode.error() != runtime::Error::NotFound) {
    return mode.error();
  }

  auto group_id = context.get_runtime_spec<int>(kSharedGroupIdOption);
  if (group_id.ok()) {
    config.group_id = group_id.get();
  } else if (group_id.error() != runtime::Error::NotFound) {
    return group_id.error();
  }

  if (config.enabled() && config.token.empty()) {
    return runtime::Error::InvalidArgument;
  }

  return config;
}

} // namespace gpu_shared
} // namespace backends
} // namespace executorch
