/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan_shared/runtime/SharedVulkanRuntimeConfig.h>

#include <cstring>

namespace executorch {
namespace backends {
namespace vulkan_shared {
namespace {

// Accept the Phase 2 option strings while making the new device terminology
// canonical. New names take precedence if both variants are supplied.
template <typename T>
runtime::Result<T> get_device_runtime_option(
    const runtime::BackendInitContext& context,
    const char* canonical,
    const char* legacy) {
  auto value = context.get_runtime_spec<T>(canonical);
  if (!value.ok() && value.error() == runtime::Error::NotFound) {
    return context.get_runtime_spec<T>(legacy);
  }
  return value;
}

runtime::Result<SharedDeviceMode> parse_device_mode(const char* value) {
  if (value == nullptr) {
    return runtime::Error::InvalidArgument;
  }
  if (std::strcmp(value, "disabled") == 0) {
    return SharedDeviceMode::kDisabled;
  }
  if (std::strcmp(value, "lookup_only") == 0) {
    return SharedDeviceMode::kLookupOnly;
  }
  if (std::strcmp(value, "lookup_or_create") == 0) {
    return SharedDeviceMode::kLookupOrCreate;
  }
  if (std::strcmp(value, "create_only") == 0) {
    return SharedDeviceMode::kCreateOnly;
  }
  return runtime::Error::InvalidArgument;
}

} // namespace

// This API is consumed by the Vulkan/VGF delegate integration follow-up PRs.
// The phase-2 runtime library intentionally has no production caller yet.
// cppcheck-suppress unusedFunction
runtime::Result<SharedVulkanRuntimeConfig> parse_shared_vulkan_runtime_config(
    const runtime::BackendInitContext& context) {
  SharedVulkanRuntimeConfig config;

  auto device_name = get_device_runtime_option<const char*>(
      context, kSharedDeviceNameOption, kLegacySharedContextNameOption);
  if (device_name.ok()) {
    config.device_name = device_name.get();
  } else if (device_name.error() != runtime::Error::NotFound) {
    return device_name.error();
  }

  auto mode = get_device_runtime_option<const char*>(
      context, kSharedDeviceModeOption, kLegacySharedContextModeOption);
  if (mode.ok()) {
    auto parsed_mode = parse_device_mode(mode.get());
    if (!parsed_mode.ok()) {
      return parsed_mode.error();
    }
    config.device_mode = parsed_mode.get();
  } else if (mode.error() != runtime::Error::NotFound) {
    return mode.error();
  }

  auto group_id = context.get_runtime_spec<int>(kSharedGroupIdOption);
  if (group_id.ok()) {
    config.group_id = group_id.get();
  } else if (group_id.error() != runtime::Error::NotFound) {
    return group_id.error();
  }

  if (config.enabled() && config.device_name.empty()) {
    return runtime::Error::InvalidArgument;
  }

  return config;
}

} // namespace vulkan_shared
} // namespace backends
} // namespace executorch
