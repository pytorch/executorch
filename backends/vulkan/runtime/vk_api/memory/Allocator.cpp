/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/vk_api/memory/Allocator.h>

namespace vkcompute {
namespace vkapi {

Allocator::Allocator(
    VkInstance instance,
    VkPhysicalDevice physical_device,
    VkDevice device)
    : instance_{},
      physical_device_(physical_device),
      device_(device),
      allocator_{VK_NULL_HANDLE} {
  VmaVulkanFunctions vk_functions{};
  vk_functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  vk_functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

  const VmaAllocatorCreateInfo allocator_create_info{
      0u, // flags
      physical_device_, // physicalDevice
      device_, // device
      0u, // preferredLargeHeapBlockSize
      nullptr, // pAllocationCallbacks
      nullptr, // pDeviceMemoryCallbacks
      nullptr, // pHeapSizeLimit
      &vk_functions, // pVulkanFunctions
      instance, // instance
      VK_API_VERSION_1_0, // vulkanApiVersion
      nullptr, // pTypeExternalMemoryHandleTypes
  };

  VK_CHECK(vmaCreateAllocator(&allocator_create_info, &allocator_));
}

Allocator::Allocator(Allocator&& other) noexcept
    : instance_(other.instance_),
      physical_device_(other.physical_device_),
      device_(other.device_),
      allocator_(other.allocator_) {
  other.allocator_ = VK_NULL_HANDLE;
  other.device_ = VK_NULL_HANDLE;
  other.physical_device_ = VK_NULL_HANDLE;
  other.instance_ = VK_NULL_HANDLE;
}

Allocator::~Allocator() {
  if (VK_NULL_HANDLE == allocator_) {
    return;
  }
  vmaDestroyAllocator(allocator_);
}

Allocation Allocator::create_allocation(
    const VkMemoryRequirements& memory_requirements,
    const VmaAllocationCreateInfo& create_info) {
  VmaAllocationCreateInfo alloc_create_info = create_info;
  // Protect against using VMA_MEMORY_USAGE_AUTO_* flags when allocating memory
  // directly, since those usage flags require that VkBufferCreateInfo and/or
  // VkImageCreateInfo also be available.
  switch (create_info.usage) {
    // The logic for the below usage options are too complex, therefore prevent
    // those from being used with direct memory allocation.
    case VMA_MEMORY_USAGE_AUTO:
    case VMA_MEMORY_USAGE_AUTO_PREFER_HOST:
      VK_THROW(
          "Only the VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE usage flag is compatible with create_allocation()");
      break;
    // Most of the time, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE will simply set the
    // DEVICE_LOCAL_BIT as a preferred memory flag. Therefore the below is a
    // decent approximation for VMA behaviour.
    case VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE:
      alloc_create_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      alloc_create_info.usage = VMA_MEMORY_USAGE_UNKNOWN;
      break;
    default:
      break;
  }

  return Allocation(allocator_, memory_requirements, alloc_create_info);
}

VulkanImage Allocator::create_image(
    const VkExtent3D& extents,
    const VkFormat image_format,
    const VkImageType image_type,
    const VkImageViewType image_view_type,
    const VulkanImage::SamplerProperties& sampler_props,
    VkSampler sampler,
    const bool allow_transfer,
    const bool allocate_memory) {
  VkImageUsageFlags usage =
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
  if (allow_transfer) {
    usage |=
        (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
  }

  VmaAllocationCreateInfo alloc_create_info = {};
  alloc_create_info.flags = DEFAULT_ALLOCATION_STRATEGY;
  alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  const VulkanImage::ImageProperties image_props{
      image_type,
      image_format,
      extents,
      usage,
  };

  const VulkanImage::ViewProperties view_props{
      image_view_type,
      image_format,
  };

  const VkImageLayout initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;

  return VulkanImage(
      allocator_,
      alloc_create_info,
      image_props,
      view_props,
      sampler_props,
      initial_layout,
      sampler,
      allocate_memory);
}

VulkanBuffer Allocator::create_staging_buffer(const VkDeviceSize size) {
  const VkBufferUsageFlags buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  VmaAllocationCreateInfo alloc_create_info = {};
  alloc_create_info.flags = DEFAULT_ALLOCATION_STRATEGY;
  alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  // Staging buffers are accessed by both the CPU and GPU, so set the
  // appropriate flags to indicate that the host device will be accessing
  // the data from this buffer.
  alloc_create_info.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
      VMA_ALLOCATION_CREATE_MAPPED_BIT;
  alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
  alloc_create_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  alloc_create_info.preferredFlags =
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

  return VulkanBuffer(allocator_, size, alloc_create_info, buffer_usage);
}

VulkanBuffer Allocator::create_storage_buffer(
    const VkDeviceSize size,
    const bool allocate_memory) {
  const VkBufferUsageFlags buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  VmaAllocationCreateInfo alloc_create_info = {};
  alloc_create_info.flags = DEFAULT_ALLOCATION_STRATEGY;
  alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  return VulkanBuffer(
      allocator_, size, alloc_create_info, buffer_usage, allocate_memory);
}

VulkanBuffer Allocator::create_uniform_buffer(const VkDeviceSize size) {
  VmaAllocationCreateInfo alloc_create_info = {};
  alloc_create_info.flags = DEFAULT_ALLOCATION_STRATEGY |
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO;

  VkBufferUsageFlags buffer_usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

  return VulkanBuffer(allocator_, size, alloc_create_info, buffer_usage);
}

} // namespace vkapi
} // namespace vkcompute
