/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/vk_api/memory/Pool.h>

#include <executorch/backends/vulkan/runtime/vk_api/Exception.h>

namespace vkcompute {
namespace vkapi {

VmaPool create_memory_pool(
    const VmaAllocator allocator,
    const uint32_t mem_type_idx,
    const size_t block_size = 0,
    const size_t max_block_count = 0) {
  VmaPoolCreateInfo create_info = {
      mem_type_idx, // memoryTypeIndex
      0u, // flags
      block_size, // blockSize
      0u, // minBlockCount
      max_block_count, // maxBlockCount
      0.0, // priority
      0u, // minAllocationAlignment
      nullptr};

  VmaPool pool = VK_NULL_HANDLE;
  VK_CHECK(vmaCreatePool(allocator, &create_info, &pool));
  return pool;
}

MemoryPool::MemoryPool()
    : allocator(VK_NULL_HANDLE),
      memory_type_idx(0u),
      block_size(0u),
      max_block_count(0u),
      handle(VK_NULL_HANDLE) {}

MemoryPool::MemoryPool(MemoryPool&& other) noexcept
    : allocator(other.allocator),
      memory_type_idx(other.memory_type_idx),
      block_size(other.block_size),
      max_block_count(other.max_block_count),
      handle(other.handle) {
  other.handle = VK_NULL_HANDLE;
}

MemoryPool& MemoryPool::operator=(MemoryPool&& other) noexcept {
  VmaAllocator tmp_allocator = allocator;
  VmaPool tmp_handle = handle;

  allocator = other.allocator;
  memory_type_idx = other.memory_type_idx;
  block_size = other.block_size;
  max_block_count = other.max_block_count;
  handle = other.handle;

  other.allocator = tmp_allocator;
  other.handle = tmp_handle;

  return *this;
}

void MemoryPool::initialize() {
  VK_CHECK_COND(handle == VK_NULL_HANDLE);
  handle = create_memory_pool(
      allocator, memory_type_idx, block_size, max_block_count);
}

MemoryPool::~MemoryPool() {
  if (handle != VK_NULL_HANDLE) {
    vmaDestroyPool(allocator, handle);
  }
}

MemoryPoolManager::MemoryPoolManager(
    VmaAllocator vma_allocator,
    const uint32_t num_memory_types)
    : allocator{vma_allocator}, memory_pools(num_memory_types) {
  for (int i = 0; i < num_memory_types; ++i) {
    memory_pools.at(i).allocator = allocator;
    memory_pools.at(i).memory_type_idx = i;
  }
}

VmaPool MemoryPoolManager::get_memory_pool(const uint32_t memory_type_idx) {
  VK_CHECK_COND(memory_type_idx < memory_pools.size());
  MemoryPool& pool = memory_pools.at(memory_type_idx);
  if (pool.handle == VK_NULL_HANDLE) {
    pool.initialize();
  }
  return pool.handle;
}

uint32_t MemoryPoolManager::get_memory_type_idx(
    const VmaAllocationCreateInfo alloc_create_info) const {
  uint32_t memory_type_idx = 0u;
  VK_CHECK(vmaFindMemoryTypeIndex(
      allocator,
      UINT32_MAX, // memoryTypeBits - using all available memory types
      &alloc_create_info,
      &memory_type_idx));
  return memory_type_idx;
}

uint32_t MemoryPoolManager::get_memory_type_idx(
    const VmaAllocationCreateInfo alloc_create_info,
    const VkImageCreateInfo image_create_info) const {
  uint32_t memory_type_idx = 0u;
  VK_CHECK(vmaFindMemoryTypeIndexForImageInfo(
      allocator, &image_create_info, &alloc_create_info, &memory_type_idx));
  return memory_type_idx;
}

uint32_t MemoryPoolManager::get_memory_type_idx(
    const VmaAllocationCreateInfo alloc_create_info,
    const VkBufferCreateInfo buffer_create_info) const {
  uint32_t memory_type_idx = 0u;
  VK_CHECK(vmaFindMemoryTypeIndexForBufferInfo(
      allocator, &buffer_create_info, &alloc_create_info, &memory_type_idx));
  return memory_type_idx;
}

} // namespace vkapi
} // namespace vkcompute
