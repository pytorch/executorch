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

struct SharedVulkanDeviceKey final {
  std::string device_name;
  int group_id = 0;

  bool valid() const {
    return !device_name.empty();
  }

  friend bool operator==(
      const SharedVulkanDeviceKey& lhs,
      const SharedVulkanDeviceKey& rhs) {
    return lhs.group_id == rhs.group_id && lhs.device_name == rhs.device_name;
  }

  friend bool operator!=(
      const SharedVulkanDeviceKey& lhs,
      const SharedVulkanDeviceKey& rhs) {
    return !(lhs == rhs);
  }
};

// These flags describe features enabled when VkDevice was created, NOT just
// features supported by VkPhysicalDevice. An externally registered device must
// explicitly declare its enabled features. Unspecified flags stay disabled.
// Values, rather than Vulkan pNext pointers, are stored across delegate lives.
struct SharedVulkanDeviceFeatures final {
  bool shader_int16 = false;
  bool shader_int64 = false;
  bool shader_float64 = false;
  bool storage_buffer_16bit_access = false;
  bool uniform_and_storage_buffer_16bit_access = false;
  bool storage_push_constant_16 = false;
  bool storage_input_output_16 = false;
  bool storage_buffer_8bit_access = false;
  bool uniform_and_storage_buffer_8bit_access = false;
  bool storage_push_constant_8 = false;
  bool shader_float16 = false;
  bool shader_int8 = false;
  bool shader_integer_dot_product = false;
  bool cooperative_matrix = false;
  bool cooperative_matrix2_workgroup_scope = false;
  bool cooperative_matrix2_flexible_dimensions = false;
  bool cooperative_matrix2_tensor_addressing = false;
  bool subgroup_size_control = false;
  bool compute_full_subgroups = false;
};

// The shared layer carries Vulkan handles but deliberately does not call Vulkan
// entry points itself. Every registered context must provide a lifetime_anchor
// whose lifetime guarantees that instance, physical_device, device, and queue
// remain valid until the final SharedVulkanDevice reference is released. The
// anchor destructor may perform backend/application Vulkan teardown.
struct SharedVulkanDeviceCreateInfo final {
  SharedVulkanDeviceKey key;
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queue_family_index = std::numeric_limits<uint32_t>::max();
  std::vector<std::string> enabled_device_extensions;
  SharedVulkanDeviceFeatures enabled_device_features;
  std::shared_ptr<void> lifetime_anchor;
};

class EXECUTORCH_VULKAN_SHARED_API SharedVulkanDevice final {
 public:
  explicit SharedVulkanDevice(SharedVulkanDeviceCreateInfo create_info);
  ~SharedVulkanDevice();

  SharedVulkanDevice(const SharedVulkanDevice&) = delete;
  SharedVulkanDevice& operator=(const SharedVulkanDevice&) = delete;

  const SharedVulkanDeviceKey& key() const;
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
  const std::vector<std::string>& enabled_device_extensions() const;
  const SharedVulkanDeviceFeatures& enabled_device_features() const;
  bool is_valid() const;

 private:
  SharedVulkanDeviceCreateInfo create_info_;
  mutable std::mutex queue_mutex_;
};

using SharedVulkanDevicePtr = std::shared_ptr<SharedVulkanDevice>;

} // namespace vulkan_shared
} // namespace backends
} // namespace executorch
