/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <executorch/backends/vulkan/runtime/vk_api/Types.h>

#include <executorch/backends/vulkan/runtime/graph/GraphConfig.h>

#include <executorch/backends/vulkan/runtime/graph/containers/Value.h>

namespace vkcompute {

class ComputeGraph;

struct SharedObject {
  friend class ComputeGraph;

  explicit SharedObject() = default;

  VkMemoryRequirements aggregate_memory_requirements;
  VmaAllocationCreateInfo aggregate_create_info;
  std::vector<ValueRef> users;
  vkapi::Allocation allocation;

  void add_user(ComputeGraph* const graph, const ValueRef idx);
  void allocate(ComputeGraph* const graph);
  void bind_users(ComputeGraph* const graph);
};

} // namespace vkcompute
