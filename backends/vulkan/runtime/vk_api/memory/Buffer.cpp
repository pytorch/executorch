/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/vk_api/memory/Buffer.h>

namespace vkcompute {
namespace vkapi {

//
// VulkanBuffer
//

VulkanBuffer::VulkanBuffer()
    : buffer_properties_{},
      allocator_(VK_NULL_HANDLE),
      memory_{},
      owns_memory_(false),
      is_copy_(false),
      handle_(VK_NULL_HANDLE) {}

VulkanBuffer::VulkanBuffer(
    VmaAllocator vma_allocator,
    const VkDeviceSize size,
    const VmaAllocationCreateInfo& allocation_create_info,
    const VkBufferUsageFlags usage,
    const bool allocate_memory)
    : buffer_properties_({
          size,
          0u,
          size,
          usage,
      }),
      allocator_(vma_allocator),
      memory_{},
      owns_memory_(allocate_memory),
      is_copy_(false),
      handle_(VK_NULL_HANDLE) {
  // If the buffer size is 0, allocate a buffer with a size of 1 byte. This is
  // to ensure that there will be some resource that can be bound to a shader.
  if (size == 0) {
    buffer_properties_.size = 1u;
    buffer_properties_.mem_range = 1u;
  }

  const VkBufferCreateInfo buffer_create_info{
      VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      buffer_properties_.size, // size
      buffer_properties_.buffer_usage, // usage
      VK_SHARING_MODE_EXCLUSIVE, // sharingMode
      0u, // queueFamilyIndexCount
      nullptr, // pQueueFamilyIndices
  };

  memory_.create_info = allocation_create_info;

  if (allocate_memory) {
    VK_CHECK(vmaCreateBuffer(
        allocator_,
        &buffer_create_info,
        &allocation_create_info,
        &handle_,
        &(memory_.allocation),
        &(memory_.allocation_info)));
  } else {
    VmaAllocatorInfo allocator_info{};
    vmaGetAllocatorInfo(allocator_, &allocator_info);
    VK_CHECK(vkCreateBuffer(
        allocator_info.device, &buffer_create_info, nullptr, &handle_));
  }
}

VulkanBuffer::VulkanBuffer(
    const VulkanBuffer& other,
    const VkDeviceSize offset,
    const VkDeviceSize range) noexcept
    : buffer_properties_(other.buffer_properties_),
      allocator_(other.allocator_),
      memory_(other.memory_),
      owns_memory_(false),
      is_copy_(true),
      handle_(other.handle_) {
  // TODO: set the offset and range appropriately
  buffer_properties_.mem_offset = other.buffer_properties_.mem_offset + offset;
  if (range != VK_WHOLE_SIZE) {
    buffer_properties_.mem_range = range;
  }
}

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept
    : buffer_properties_(other.buffer_properties_),
      allocator_(other.allocator_),
      memory_(std::move(other.memory_)),
      owns_memory_(other.owns_memory_),
      is_copy_(other.is_copy_),
      handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
}

VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& other) noexcept {
  VkBuffer tmp_buffer = handle_;
  bool tmp_owns_memory = owns_memory_;

  buffer_properties_ = other.buffer_properties_;
  allocator_ = other.allocator_;
  memory_ = std::move(other.memory_);
  owns_memory_ = other.owns_memory_;
  is_copy_ = other.is_copy_;
  handle_ = other.handle_;

  other.handle_ = tmp_buffer;
  other.owns_memory_ = tmp_owns_memory;

  return *this;
}

VulkanBuffer::~VulkanBuffer() {
  // Do not destroy the VkBuffer if this class instance is a copy of another
  // class instance, since this means that this class instance does not have
  // ownership of the underlying resource.
  if (VK_NULL_HANDLE != handle_ && !is_copy_) {
    if (owns_memory_) {
      vmaDestroyBuffer(allocator_, handle_, memory_.allocation);
    } else {
      vkDestroyBuffer(this->device(), handle_, nullptr);
    }
    // Prevent the underlying memory allocation from being freed; it was either
    // freed by vmaDestroyBuffer, or this resource does not own the underlying
    // memory
    memory_.allocation = VK_NULL_HANDLE;
  }
}

VkMemoryRequirements VulkanBuffer::get_memory_requirements() const {
  VkMemoryRequirements memory_requirements;
  vkGetBufferMemoryRequirements(this->device(), handle_, &memory_requirements);
  return memory_requirements;
}

//
// MemoryMap
//

MemoryMap::MemoryMap(const VulkanBuffer& buffer, const uint8_t access)
    : access_(access),
      allocator_(buffer.vma_allocator()),
      allocation_(buffer.allocation()),
      data_(nullptr),
      data_len_{buffer.mem_size()} {
  if (allocation_) {
    VK_CHECK(vmaMapMemory(allocator_, allocation_, &data_));
  }
}

MemoryMap::MemoryMap(MemoryMap&& other) noexcept
    : access_(other.access_),
      allocator_(other.allocator_),
      allocation_(other.allocation_),
      data_(other.data_),
      data_len_{other.data_len_} {
  other.allocation_ = VK_NULL_HANDLE;
  other.data_ = nullptr;
}

MemoryMap::~MemoryMap() {
  if (!data_) {
    return;
  }

  if (allocation_) {
    if (access_ & MemoryAccessType::WRITE) {
      // Call will be ignored by implementation if the memory type this
      // allocation belongs to is not HOST_VISIBLE or is HOST_COHERENT, which is
      // the behavior we want. Don't check the result here as the destructor
      // cannot throw.
      vmaFlushAllocation(allocator_, allocation_, 0u, VK_WHOLE_SIZE);
    }

    vmaUnmapMemory(allocator_, allocation_);
  }
}

void MemoryMap::invalidate() {
  if (access_ & MemoryAccessType::READ && allocation_) {
    // Call will be ignored by implementation if the memory type this allocation
    // belongs to is not HOST_VISIBLE or is HOST_COHERENT, which is the behavior
    // we want.
    VK_CHECK(
        vmaInvalidateAllocation(allocator_, allocation_, 0u, VK_WHOLE_SIZE));
  }
}

//
// BufferMemoryBarrier
//

BufferMemoryBarrier::BufferMemoryBarrier(
    const VkAccessFlags src_access_flags,
    const VkAccessFlags dst_access_flags,
    const VulkanBuffer& buffer)
    : handle{
          VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, // sType
          nullptr, // pNext
          src_access_flags, // srcAccessMask
          dst_access_flags, // dstAccessMask
          VK_QUEUE_FAMILY_IGNORED, // srcQueueFamilyIndex
          VK_QUEUE_FAMILY_IGNORED, // dstQueueFamilyIndex
          buffer.handle_, // buffer
          buffer.buffer_properties_.mem_offset, // offset
          buffer.buffer_properties_.mem_range, // size
      } {}

} // namespace vkapi
} // namespace vkcompute
