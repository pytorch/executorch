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
      allocation(VK_NULL_HANDLE),
      allocation_info({}),
      is_copy_(false) {}

Allocation::Allocation(
    VmaAllocator vma_allocator,
    const VkMemoryRequirements& mem_props,
    const VmaAllocationCreateInfo& create_info)
    : memory_requirements(mem_props),
      create_info(create_info),
      allocator(vma_allocator),
      allocation(VK_NULL_HANDLE),
      allocation_info({}),
      is_copy_(false) {
  VK_CHECK(vmaAllocateMemory(
      allocator, &memory_requirements, &create_info, &allocation, nullptr));
}

Allocation::Allocation(const Allocation& other) noexcept
    : memory_requirements(other.memory_requirements),
      create_info(other.create_info),
      allocator(other.allocator),
      allocation(other.allocation),
      allocation_info(other.allocation_info),
      is_copy_(true) {}

Allocation::Allocation(Allocation&& other) noexcept
    : memory_requirements(other.memory_requirements),
      create_info(other.create_info),
      allocator(other.allocator),
      allocation(other.allocation),
      allocation_info(other.allocation_info),
      is_copy_(other.is_copy_) {
  other.allocation = VK_NULL_HANDLE;
  other.allocation_info = {};
}

Allocation& Allocation::operator=(Allocation&& other) noexcept {
  VmaAllocation tmp_allocation = allocation;

  memory_requirements = other.memory_requirements;
  create_info = other.create_info;
  allocator = other.allocator;
  allocation = other.allocation;
  allocation_info = other.allocation_info;
  is_copy_ = other.is_copy_;

  other.allocation = tmp_allocation;
  other.allocation_info = {};

  return *this;
}

Allocation::~Allocation() {
  // Do not destroy the VmaAllocation if this class instance is a copy of some
  // other class instance, since this means that this class instance does not
  // have ownership of the underlying resource.
  if (VK_NULL_HANDLE != allocation && !is_copy_) {
    vmaFreeMemory(allocator, allocation);
  }
}

} // namespace vkapi
} // namespace vkcompute
