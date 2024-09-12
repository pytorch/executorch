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

// Forward declare vTensor classes such that they can be set as friend classes
namespace api {
class vTensorStorage;
} // namespace api

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

 protected:
  /*
   * The Copy constructor and allows for creation of a class instance that are
   * "aliases" of another class instance. The resulting class instance will not
   * have ownership of the underlying VkBuffer.
   *
   * This behaviour is analogous to creating a copy of a pointer, thus it is
   * unsafe, as the original class instance may be destroyed before the copy.
   * These constructors are therefore marked protected so that they may be used
   * only in situations where the lifetime of the original class instance is
   * guaranteed to exceed, or at least be the same as, the lifetime of the
   * copied class instance.
   */
  VulkanBuffer(
      const VulkanBuffer& other,
      const VkDeviceSize offset = 0u,
      const VkDeviceSize range = VK_WHOLE_SIZE) noexcept;

 public:
  // To discourage creating copies, the assignment operator is still deleted.
  VulkanBuffer& operator=(const VulkanBuffer& other) = delete;

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
  // Indicates whether this VulkanBuffer was copied from another VulkanBuffer,
  // thus it does not have ownership of the underlying VKBuffer
  bool is_copy_;
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

  inline VmaAllocationInfo allocation_info() const {
    return memory_.allocation_info;
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

  inline bool is_copy() const {
    return is_copy_;
  }

  operator bool() const {
    return (handle_ != VK_NULL_HANDLE);
  }

  inline bool is_copy_of(const VulkanBuffer& other) const {
    return (handle_ == other.handle_) && is_copy_;
  }

  inline void bind_allocation(const Allocation& memory) {
    VK_CHECK_COND(!memory_, "Cannot bind an already bound allocation!");
    VK_CHECK(vmaBindBufferMemory(allocator_, memory.allocation, handle_));
    memory_.allocation = memory.allocation;
  }

  VkMemoryRequirements get_memory_requirements() const;

  friend class api::vTensorStorage;
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
