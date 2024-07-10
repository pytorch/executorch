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

#include <executorch/backends/vulkan/runtime/vk_api/memory/vma_api.h>

#include <ostream>

std::ostream& operator<<(std::ostream& out, VmaTotalStatistics stats);

namespace vkcompute {
namespace vkapi {

struct Allocation final {
  explicit Allocation();

  explicit Allocation(
      const VmaAllocator,
      const VkMemoryRequirements&,
      const VmaAllocationCreateInfo&);

  Allocation(const Allocation&) = delete;
  Allocation& operator=(const Allocation&) = delete;

  Allocation(Allocation&&) noexcept;
  Allocation& operator=(Allocation&&) noexcept;

  ~Allocation();

  VkMemoryRequirements memory_requirements;
  // The properties this allocation was created with
  VmaAllocationCreateInfo create_info;
  // The allocator object this was allocated from
  VmaAllocator allocator;
  // Handles to the allocated memory
  VmaAllocation allocation;

  operator bool() const {
    return (allocation != VK_NULL_HANDLE);
  }
};

} // namespace vkapi
} // namespace vkcompute
