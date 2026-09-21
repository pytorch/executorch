/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan_shared/runtime/export.h>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace executorch {
namespace backends {
namespace vulkan_shared {

struct SharedVulkanContextKey final {
  std::string context_name;
  int group_id = 0;

  bool valid() const {
    return !context_name.empty();
  }

  friend bool operator==(
      const SharedVulkanContextKey& lhs,
      const SharedVulkanContextKey& rhs) {
    return lhs.group_id == rhs.group_id && lhs.context_name == rhs.context_name;
  }

  friend bool operator!=(
      const SharedVulkanContextKey& lhs,
      const SharedVulkanContextKey& rhs) {
    return !(lhs == rhs);
  }
};

// The shared layer carries Vulkan handles but deliberately does not call Vulkan
// entry points itself. Every registered context must provide a lifetime_anchor
// whose lifetime guarantees that instance, physical_device, device, and queue
// remain valid until the final SharedVulkanContext reference is released. The
// anchor destructor may perform backend/application Vulkan teardown.
struct SharedVulkanContextCreateInfo final {
  SharedVulkanContextKey key;
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queue_family_index = std::numeric_limits<uint32_t>::max();
  std::vector<std::string> enabled_device_extensions;
  std::shared_ptr<void> lifetime_anchor;
};

class EXECUTORCH_VULKAN_SHARED_API SharedVulkanContext final {
 public:
  explicit SharedVulkanContext(SharedVulkanContextCreateInfo create_info);
  ~SharedVulkanContext();

  SharedVulkanContext(const SharedVulkanContext&) = delete;
  SharedVulkanContext& operator=(const SharedVulkanContext&) = delete;

  const SharedVulkanContextKey& key() const;
  VkInstance instance() const;
  VkPhysicalDevice physical_device() const;
  VkDevice device() const;

  // Vulkan queue operations require external host synchronization. All
  // delegates sharing this context must issue queue operations through this
  // callback so they synchronize on the same mutex. The VkQueue must not be
  // retained and used after the callback returns.
  template <typename Fn>
  decltype(auto) with_locked_queue(Fn&& fn) const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return std::forward<Fn>(fn)(create_info_.queue);
  }

  uint32_t queue_family_index() const;
  bool has_device_extension(std::string_view extension_name) const;
  bool is_valid() const;

 private:
  SharedVulkanContextCreateInfo create_info_;
  mutable std::mutex queue_mutex_;
};

using SharedVulkanContextPtr = std::shared_ptr<SharedVulkanContext>;

} // namespace vulkan_shared
} // namespace backends
} // namespace executorch
