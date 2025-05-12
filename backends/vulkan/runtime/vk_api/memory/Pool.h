/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#include <executorch/backends/vulkan/runtime/vk_api/memory/vma_api.h>
#include <executorch/backends/vulkan/runtime/vk_api/vk_api.h>

#include <vector>

namespace vkcompute {
namespace vkapi {

struct MemoryPool final {
  MemoryPool();

  MemoryPool(const MemoryPool& other) = delete;
  MemoryPool& operator=(const MemoryPool& other) = delete;

  MemoryPool(MemoryPool&& other) noexcept;
  MemoryPool& operator=(MemoryPool&& other) noexcept;

  ~MemoryPool();

  void initialize();

  VmaAllocator allocator;
  uint32_t memory_type_idx;
  size_t block_size;
  size_t max_block_count;
  VmaPool handle;
};

class MemoryPoolManager final {
 public:
  explicit MemoryPoolManager(
      VmaAllocator vma_allocator,
      const uint32_t num_memory_types);

  VmaPool get_memory_pool(const uint32_t memory_type_idx);

  uint32_t get_memory_type_idx(
      const VmaAllocationCreateInfo alloc_create_info) const;

  uint32_t get_memory_type_idx(
      const VmaAllocationCreateInfo alloc_create_info,
      const VkImageCreateInfo image_create_info) const;

  uint32_t get_memory_type_idx(
      const VmaAllocationCreateInfo alloc_create_info,
      const VkBufferCreateInfo buffer_create_info) const;

 private:
  VmaAllocator allocator;
  std::vector<MemoryPool> memory_pools;
};

} // namespace vkapi
} // namespace vkcompute
