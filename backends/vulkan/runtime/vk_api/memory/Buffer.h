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

#include <executorch/backends/vulkan/runtime/utils/VecUtils.h>

#include <executorch/backends/vulkan/runtime/vk_api/memory/vma_api.h>

#include <executorch/backends/vulkan/runtime/vk_api/memory/Allocation.h>

namespace vkcompute {
namespace vkapi {

using MemoryAccessFlags = uint8_t;

enum MemoryAccessType : MemoryAccessFlags {
  NONE = 0u << 0u,
  READ = 1u << 0u,
  WRITE = 1u << 1u,
};

class VulkanBuffer final {
 public:
  struct BufferProperties final {
    VkDeviceSize size;
    VkDeviceSize mem_offset;
    VkDeviceSize mem_range;
    VkBufferUsageFlags buffer_usage;
  };

  explicit VulkanBuffer();

  explicit VulkanBuffer(
      const VmaAllocator,
      const VkDeviceSize,
      const VmaAllocationCreateInfo&,
      const VkBufferUsageFlags,
      const bool allocate_memory = true);

  VulkanBuffer(const VulkanBuffer&) = delete;
  VulkanBuffer& operator=(const VulkanBuffer&) = delete;

  VulkanBuffer(VulkanBuffer&&) noexcept;
  VulkanBuffer& operator=(VulkanBuffer&&) noexcept;

  ~VulkanBuffer();

  struct Package final {
    VkBuffer handle;
    VkDeviceSize buffer_offset;
    VkDeviceSize buffer_range;
  };

  friend struct BufferMemoryBarrier;

 private:
  BufferProperties buffer_properties_;
  VmaAllocator allocator_;
  Allocation memory_;
  // Indicates whether the underlying memory is owned by this resource
  bool owns_memory_;
  VkBuffer handle_;

 public:
  inline VkDevice device() const {
    VmaAllocatorInfo allocator_info{};
    vmaGetAllocatorInfo(allocator_, &allocator_info);
    return allocator_info.device;
  }

  inline VmaAllocator vma_allocator() const {
    return allocator_;
  }

  inline VmaAllocation allocation() const {
    return memory_.allocation;
  }

  inline VmaAllocationCreateInfo allocation_create_info() const {
    return VmaAllocationCreateInfo(memory_.create_info);
  }

  inline VkBuffer handle() const {
    return handle_;
  }

  inline VkDeviceSize mem_offset() const {
    return buffer_properties_.mem_offset;
  }

  inline VkDeviceSize mem_range() const {
    return buffer_properties_.mem_range;
  }

  inline VkDeviceSize mem_size() const {
    return buffer_properties_.size;
  }

  inline bool has_memory() const {
    return (memory_.allocation != VK_NULL_HANDLE);
  }

  inline bool owns_memory() const {
    return owns_memory_;
  }

  operator bool() const {
    return (handle_ != VK_NULL_HANDLE);
  }

  inline void bind_allocation(const Allocation& memory) {
    VK_CHECK_COND(!memory_, "Cannot bind an already bound allocation!");
    VK_CHECK(vmaBindBufferMemory(allocator_, memory.allocation, handle_));
    memory_.allocation = memory.allocation;
  }

  VkMemoryRequirements get_memory_requirements() const;
};

class MemoryMap final {
 public:
  explicit MemoryMap(
      const VulkanBuffer& buffer,
      const MemoryAccessFlags access);

  MemoryMap(const MemoryMap&) = delete;
  MemoryMap& operator=(const MemoryMap&) = delete;

  MemoryMap(MemoryMap&&) noexcept;
  MemoryMap& operator=(MemoryMap&&) = delete;

  ~MemoryMap();

 private:
  uint8_t access_;
  VmaAllocator allocator_;
  VmaAllocation allocation_;
  void* data_;
  VkDeviceSize data_len_;

 public:
  template <typename T>
  T* data() {
    return reinterpret_cast<T*>(data_);
  }

  inline size_t nbytes() {
    return utils::safe_downcast<size_t>(data_len_);
  }

  void invalidate();
};

struct BufferMemoryBarrier final {
  VkBufferMemoryBarrier handle;

  BufferMemoryBarrier(
      const VkAccessFlags src_access_flags,
      const VkAccessFlags dst_access_flags,
      const VulkanBuffer& buffer);
};

} // namespace vkapi
} // namespace vkcompute
