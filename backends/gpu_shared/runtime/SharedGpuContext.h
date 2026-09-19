/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/gpu_shared/runtime/export.h>

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
namespace gpu_shared {

struct SharedGpuContextKey final {
  std::string token;
  int group_id = 0;

  bool valid() const {
    return !token.empty();
  }

  friend bool operator==(
      const SharedGpuContextKey& lhs,
      const SharedGpuContextKey& rhs) {
    return lhs.group_id == rhs.group_id && lhs.token == rhs.token;
  }

  friend bool operator!=(
      const SharedGpuContextKey& lhs,
      const SharedGpuContextKey& rhs) {
    return !(lhs == rhs);
  }
};

// The shared layer carries Vulkan handles but deliberately does not call Vulkan
// entry points itself. Every registered context must provide a lifetime_anchor
// whose lifetime guarantees that instance, physical_device, device, and queue
// remain valid until the final SharedGpuContext reference is released. The
// anchor destructor may perform backend/application Vulkan teardown.
struct SharedGpuContextCreateInfo final {
  SharedGpuContextKey key;
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queue_family_index = std::numeric_limits<uint32_t>::max();
  std::vector<std::string> enabled_device_extensions;
  std::shared_ptr<void> lifetime_anchor;
};

class EXECUTORCH_GPU_SHARED_API SharedGpuContext final {
 public:
  explicit SharedGpuContext(SharedGpuContextCreateInfo create_info);
  ~SharedGpuContext();

  SharedGpuContext(const SharedGpuContext&) = delete;
  SharedGpuContext& operator=(const SharedGpuContext&) = delete;

  const SharedGpuContextKey& key() const;
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
  SharedGpuContextCreateInfo create_info_;
  mutable std::mutex queue_mutex_;
};

using SharedGpuContextPtr = std::shared_ptr<SharedGpuContext>;

} // namespace gpu_shared
} // namespace backends
} // namespace executorch
