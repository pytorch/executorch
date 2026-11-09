/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan_shared/runtime/export.h>
#include <executorch/runtime/backend/backend_init_context.h>
#include <executorch/runtime/core/result.h>

#include <cstdint>
#include <string>

namespace executorch {
namespace backends {
namespace vulkan_shared {

inline constexpr char kSharedDeviceNameOption[] = "vulkan_shared_device_name";
inline constexpr char kSharedDeviceModeOption[] = "vulkan_shared_device_mode";
inline constexpr char kLegacySharedContextNameOption[] =
    "vulkan_shared_context_name";
inline constexpr char kLegacySharedContextModeOption[] =
    "vulkan_shared_context_mode";
inline constexpr char kSharedGroupIdOption[] = "vulkan_shared_group_id";

enum class SharedDeviceMode : uint8_t {
  kDisabled = 0,
  kLookupOnly = 1,
  kLookupOrCreate = 2,
  kCreateOnly = 3,
};

// Load-time configuration shared by the VGF and Vulkan delegates. These values
// are RuntimeSpec options: context selection is a deployment concern and must
// not be serialized into a backend CompileSpec or a .pte file.
struct SharedVulkanRuntimeConfig final {
  std::string device_name = "default";
  int group_id = 0;
  SharedDeviceMode device_mode = SharedDeviceMode::kLookupOrCreate;

  bool enabled() const {
    return device_mode != SharedDeviceMode::kDisabled;
  }

  bool lookup_only() const {
    return device_mode == SharedDeviceMode::kLookupOnly;
  }

  bool lookup_or_create() const {
    return device_mode == SharedDeviceMode::kLookupOrCreate;
  }

  bool create_only() const {
    return device_mode == SharedDeviceMode::kCreateOnly;
  }
};

EXECUTORCH_VULKAN_SHARED_API runtime::Result<SharedVulkanRuntimeConfig>
parse_shared_vulkan_runtime_config(const runtime::BackendInitContext& context);

} // namespace vulkan_shared
} // namespace backends
} // namespace executorch
