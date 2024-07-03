/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/containers/SharedObject.h>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

namespace vkcompute {

void SharedObject::add_user(ComputeGraph* const graph, const ValueRef idx) {
  vTensorPtr t = graph->get_tensor(idx);

  //
  // Aggregate Memory Requirements
  //

  const VkMemoryRequirements mem_reqs = t->get_memory_requirements();
  aggregate_memory_requirements.size =
      std::max(mem_reqs.size, aggregate_memory_requirements.size);
  aggregate_memory_requirements.alignment =
      std::max(mem_reqs.alignment, aggregate_memory_requirements.alignment);
  aggregate_memory_requirements.memoryTypeBits |= mem_reqs.memoryTypeBits;

  //
  // Aggregate Allocation Create Info
  //

  const VmaAllocationCreateInfo create_info = t->get_allocation_create_info();
  // Clear out CREATE_STRATEGY bit flags in case of conflict
  VmaAllocationCreateFlags clear_mask = ~VMA_ALLOCATION_CREATE_STRATEGY_MASK;
  VmaAllocationCreateFlags create_flags = create_info.flags & clear_mask;
  // Use the default allocation strategy
  aggregate_create_info.flags =
      create_flags | vkapi::DEFAULT_ALLOCATION_STRATEGY;

  // Set the usage flag if it is currently not set
  if (aggregate_create_info.usage == VMA_MEMORY_USAGE_UNKNOWN) {
    aggregate_create_info.usage = create_info.usage;
  }
  // Otherwise check that there is no conflict regarding usage
  VK_CHECK_COND(aggregate_create_info.usage == create_info.usage);
  aggregate_create_info.requiredFlags |= create_info.requiredFlags;
  aggregate_create_info.preferredFlags |= create_info.preferredFlags;

  users.emplace_back(idx);
}

void SharedObject::allocate(ComputeGraph* const graph) {
  if (aggregate_memory_requirements.size == 0) {
    return;
  }
  allocation = graph->context()->adapter_ptr()->vma().create_allocation(
      aggregate_memory_requirements, aggregate_create_info);
}

void SharedObject::bind_users(ComputeGraph* const graph) {
  if (users.empty()) {
    return;
  }
  for (const ValueRef idx : users) {
    graph->get_tensor(idx)->bind_allocation(allocation);
  }
}

} // namespace vkcompute
