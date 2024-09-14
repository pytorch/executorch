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

 protected:
  /*
   * The Copy constructor allows for creation of a class instance that are
   * "aliases" of another class instance. The resulting class instance will not
   * have ownership of the underlying VmaAllocation.
   *
   * This behaviour is analogous to creating a copy of a pointer, thus it is
   * unsafe, as the original class instance may be destroyed before the copy.
   * These constructors are therefore marked protected so that they may be used
   * only in situations where the lifetime of the original class instance is
   * guaranteed to exceed, or at least be the same as, the lifetime of the
   * copied class instance.
   */
  Allocation(const Allocation&) noexcept;

 public:
  // To discourage creating copies, the assignment operator is still deleted.
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
  // Information about the allocated memory
  VmaAllocationInfo allocation_info;

 private:
  // Indicates whether this class instance is a copy of another class instance,
  // in which case it does not have ownership of the underlying VmaAllocation
  bool is_copy_;

 public:
  operator bool() const {
    return (allocation != VK_NULL_HANDLE);
  }

  inline bool is_copy() const {
    return is_copy_;
  }

  friend class VulkanBuffer;
  friend class VulkanImage;
};

} // namespace vkapi
} // namespace vkcompute
