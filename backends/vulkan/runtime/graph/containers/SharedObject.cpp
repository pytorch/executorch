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

bool SharedObject::has_user(const ValueRef idx) const {
  return std::find(users.begin(), users.end(), idx) != users.end();
}

void SharedObject::add_user(ComputeGraph* const graph, const ValueRef idx) {
  vTensorPtr t = graph->get_tensor(idx);

  // Aggregate Memory Requirements
  const VkMemoryRequirements mem_reqs = t->get_memory_requirements();
  aggregate_memory_requirements.size =
      std::max(mem_reqs.size, aggregate_memory_requirements.size);
  aggregate_memory_requirements.alignment =
      std::max(mem_reqs.alignment, aggregate_memory_requirements.alignment);
  aggregate_memory_requirements.memoryTypeBits |= mem_reqs.memoryTypeBits;

  users.emplace_back(idx);
}

void SharedObject::allocate(ComputeGraph* const graph) {
  if (aggregate_memory_requirements.size == 0) {
    return;
  }

  VmaAllocationCreateInfo alloc_create_info =
      graph->context()->adapter_ptr()->vma().gpuonly_resource_create_info();

  allocation = graph->context()->adapter_ptr()->vma().create_allocation(
      aggregate_memory_requirements, alloc_create_info);
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
