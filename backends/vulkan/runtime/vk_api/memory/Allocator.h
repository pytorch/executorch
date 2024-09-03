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
#include <executorch/backends/vulkan/runtime/vk_api/memory/Buffer.h>
#include <executorch/backends/vulkan/runtime/vk_api/memory/Image.h>

namespace vkcompute {
namespace vkapi {

constexpr VmaAllocationCreateFlags DEFAULT_ALLOCATION_STRATEGY =
    VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT;

class Allocator final {
 public:
  explicit Allocator(
      VkInstance instance,
      VkPhysicalDevice physical_device,
      VkDevice device);

  Allocator(const Allocator&) = delete;
  Allocator& operator=(const Allocator&) = delete;

  Allocator(Allocator&&) noexcept;
  Allocator& operator=(Allocator&&) = delete;

  ~Allocator();

 private:
  VkInstance instance_;
  VkPhysicalDevice physical_device_;
  VkDevice device_;
  VmaAllocator allocator_;

 public:
  Allocation create_allocation(
      const VkMemoryRequirements& memory_requirements,
      const VmaAllocationCreateInfo& create_info);

  VulkanImage create_image(
      const VkExtent3D&,
      const VkFormat,
      const VkImageType,
      const VkImageViewType,
      const VulkanImage::SamplerProperties&,
      VkSampler,
      const bool allow_transfer = false,
      const bool allocate_memory = true);

  VulkanBuffer create_staging_buffer(const VkDeviceSize);

  VulkanBuffer create_storage_buffer(
      const VkDeviceSize,
      const bool allocate_memory = true);

  /*
   * Create a uniform buffer with a specified size
   */
  VulkanBuffer create_uniform_buffer(const VkDeviceSize);

  /*
   * Create a uniform buffer containing the data in an arbitrary struct
   */
  template <typename Block>
  VulkanBuffer create_params_buffer(const Block& block);

  VmaTotalStatistics get_memory_statistics() const {
    VmaTotalStatistics stats = {};
    vmaCalculateStatistics(allocator_, &stats);
    return stats;
  }
};

//
// Impl
//

template <typename Block>
inline VulkanBuffer Allocator::create_params_buffer(const Block& block) {
  VulkanBuffer uniform_buffer = create_uniform_buffer(sizeof(Block));

  // Fill the uniform buffer with data in block
  {
    MemoryMap mapping(uniform_buffer, MemoryAccessType::WRITE);
    Block* data_ptr = mapping.template data<Block>();

    *data_ptr = block;
  }

  return uniform_buffer;
}

} // namespace vkapi
} // namespace vkcompute
