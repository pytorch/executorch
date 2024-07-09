/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/vk_api/memory/Allocation.h>

#define PRINT_FIELD(struct, field) #field << ": " << struct.field << std::endl

std::ostream& operator<<(std::ostream& out, VmaTotalStatistics stats) {
  VmaDetailedStatistics total_stats = stats.total;
  out << "VmaTotalStatistics: " << std::endl;
  out << "  " << PRINT_FIELD(total_stats.statistics, blockCount);
  out << "  " << PRINT_FIELD(total_stats.statistics, allocationCount);
  out << "  " << PRINT_FIELD(total_stats.statistics, blockBytes);
  out << "  " << PRINT_FIELD(total_stats.statistics, allocationBytes);
  return out;
}

#undef PRINT_FIELD

namespace vkcompute {
namespace vkapi {

Allocation::Allocation()
    : memory_requirements{},
      create_info{},
      allocator(VK_NULL_HANDLE),
      allocation(VK_NULL_HANDLE) {}

Allocation::Allocation(
    VmaAllocator vma_allocator,
    const VkMemoryRequirements& mem_props,
    const VmaAllocationCreateInfo& create_info)
    : memory_requirements(mem_props),
      create_info(create_info),
      allocator(vma_allocator),
      allocation(VK_NULL_HANDLE) {
  VK_CHECK(vmaAllocateMemory(
      allocator, &memory_requirements, &create_info, &allocation, nullptr));
}

Allocation::Allocation(Allocation&& other) noexcept
    : memory_requirements(other.memory_requirements),
      create_info(other.create_info),
      allocator(other.allocator),
      allocation(other.allocation) {
  other.allocation = VK_NULL_HANDLE;
}

Allocation& Allocation::operator=(Allocation&& other) noexcept {
  VmaAllocation tmp_allocation = allocation;

  memory_requirements = other.memory_requirements;
  create_info = other.create_info;
  allocator = other.allocator;
  allocation = other.allocation;

  other.allocation = tmp_allocation;

  return *this;
}

Allocation::~Allocation() {
  if (VK_NULL_HANDLE != allocation) {
    vmaFreeMemory(allocator, allocation);
  }
}

} // namespace vkapi
} // namespace vkcompute
