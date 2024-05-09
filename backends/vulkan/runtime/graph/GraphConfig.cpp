/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/GraphConfig.h>

namespace vkcompute {

GraphConfig::GraphConfig() {
  // No automatic submissions
  const uint32_t cmd_submit_frequency = UINT32_MAX;

  // Only one command buffer will be encoded at a time
  const api::CommandPoolConfig cmd_config{
      1u, // cmd_pool_initial_size
      1u, // cmd_pool_batch_size
  };

  // Use lazy descriptor pool initialization by default; the graph runtime will
  // tally up the number of descriptor sets needed while building the graph and
  // trigger descriptor pool initialization with exact sizes before encoding the
  // command buffer.
  const api::DescriptorPoolConfig descriptor_pool_config{
      0u, // descriptor_pool_max_sets
      0u, // descriptor_uniform_buffer_count
      0u, // descriptor_storage_buffer_count
      0u, // descriptor_combined_sampler_count
      0u, // descriptor_storage_image_count
      0u, // descriptor_pile_sizes
  };

  const api::QueryPoolConfig query_pool_config{};

  context_config = {
      cmd_submit_frequency,
      cmd_config,
      descriptor_pool_config,
      query_pool_config,
  };

  // Empirically selected safety factor. If descriptor pools start running out
  // of memory, increase this safety factor.
  descriptor_pool_safety_factor = 1.25;

  // For now, force kTexture3D storage as we are still developing shader support
  // for buffer storage type.
  enable_storage_type_override = true;
  storage_type_override = api::kTexture3D;

  // For now, force kWidthPacked memory layout by default as we are still
  // developing support for other memory layouts. In the future memory layout
  // settings will be serialized as part of the graph.
  enable_memory_layout_override = true;
  memory_layout_override = api::kWidthPacked;
}

void GraphConfig::set_storage_type_override(api::StorageType storage_type) {
  enable_storage_type_override = true;
  storage_type_override = storage_type;
}

void GraphConfig::set_memory_layout_override(
    api::GPUMemoryLayout memory_layout) {
  enable_memory_layout_override = true;
  memory_layout_override = memory_layout;
}

} // namespace vkcompute
