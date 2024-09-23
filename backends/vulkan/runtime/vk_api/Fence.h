/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#include <executorch/backends/vulkan/runtime/vk_api/vk_api.h>

#include <executorch/backends/vulkan/runtime/vk_api/Exception.h>

#include <stack>

namespace vkcompute {
namespace vkapi {

class VulkanFence final {
 public:
  // TODO: This is required for the lazy allocation pattern in api::vTensor.
  //       It will be disabled pending future refactors.
  explicit VulkanFence();

  explicit VulkanFence(VkDevice);

  VulkanFence(const VulkanFence&) = delete;
  VulkanFence& operator=(const VulkanFence&) = delete;

  VulkanFence(VulkanFence&&) noexcept;
  VulkanFence& operator=(VulkanFence&&) noexcept;

  ~VulkanFence();

 private:
  VkDevice device_;
  VkFence handle_;
  bool waiting_;

 public:
  // Used to get the handle for a queue submission.
  VkFence get_submit_handle() {
    if (handle_ != VK_NULL_HANDLE) {
      // Indicate we are now waiting for this fence to be signaled
      waiting_ = true;
    }
    return handle_;
  }

  VkFence handle() {
    return handle_;
  }

  // Trigger a synchronous wait for the fence to be signaled
  void wait();

  bool waiting() const {
    return waiting_;
  }

  operator bool() const {
    return (VK_NULL_HANDLE != handle_);
  }
};

// A pool to track created Fences and reuse ones that are available.
// Only intended to be modified by one thread at a time.
struct FencePool final {
  VkDevice device_;

  std::stack<VulkanFence> pool_;

  explicit FencePool(VkDevice device) : device_(device), pool_{} {}

  // Returns an rvalue reference to a fence, so that it can be moved
  inline VulkanFence get_fence() {
    if (pool_.empty()) {
      VulkanFence new_fence = VulkanFence(device_);
      return new_fence;
    }

    VulkanFence top_fence = std::move(pool_.top());
    pool_.pop();

    return top_fence;
  }

  // Marks the fence as available
  inline void return_fence(VulkanFence& fence) {
    pool_.push(std::move(fence));
  }
};

} // namespace vkapi
} // namespace vkcompute
