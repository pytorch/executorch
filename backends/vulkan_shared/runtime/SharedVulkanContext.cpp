/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan_shared/runtime/SharedVulkanContext.h>

#include <algorithm>
#include <utility>

namespace executorch {
namespace backends {
namespace vulkan_shared {

SharedVulkanContext::SharedVulkanContext(
    SharedVulkanContextCreateInfo create_info)
    : create_info_(std::move(create_info)) {}

SharedVulkanContext::~SharedVulkanContext() = default;

const SharedVulkanContextKey& SharedVulkanContext::key() const {
  return create_info_.key;
}

VkInstance SharedVulkanContext::instance() const {
  return create_info_.instance;
}

VkPhysicalDevice SharedVulkanContext::physical_device() const {
  return create_info_.physical_device;
}

VkDevice SharedVulkanContext::device() const {
  return create_info_.device;
}

uint32_t SharedVulkanContext::queue_family_index() const {
  return create_info_.queue_family_index;
}

bool SharedVulkanContext::has_device_extension(
    std::string_view extension_name) const {
  return std::any_of(
      create_info_.enabled_device_extensions.begin(),
      create_info_.enabled_device_extensions.end(),
      [extension_name](const std::string& enabled_extension) {
        return enabled_extension == extension_name;
      });
}

bool SharedVulkanContext::is_valid() const {
  return create_info_.key.valid() && create_info_.instance != VK_NULL_HANDLE &&
      create_info_.physical_device != VK_NULL_HANDLE &&
      create_info_.device != VK_NULL_HANDLE &&
      create_info_.queue != VK_NULL_HANDLE &&
      create_info_.queue_family_index != std::numeric_limits<uint32_t>::max() &&
      create_info_.lifetime_anchor != nullptr;
}

} // namespace vulkan_shared
} // namespace backends
} // namespace executorch
