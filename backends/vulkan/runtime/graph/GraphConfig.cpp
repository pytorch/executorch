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
  const uint32_t submit_frequency = UINT32_MAX;

  // Only one command buffer will be encoded at a time
  const api::CommandPoolConfig cmd_config{
      1u, // cmdPoolInitialSize
      1u, // cmdPoolBatchSize
  };

  // Use lazy descriptor pool initialization by default; the graph runtime will
  // tally up the number of descriptor sets needed while building the graph and
  // trigger descriptor pool initialization with exact sizes before encoding the
  // command buffer.
  const api::DescriptorPoolConfig descriptor_pool_config{
      0u, // descriptorPoolMaxSets
      0u, // descriptorUniformBufferCount
      0u, // descriptorStorageBufferCount
      0u, // descriptorCombinedSamplerCount
      0u, // descriptorStorageImageCount
      0u, // descriptorPileSizes
  };

  const api::QueryPoolConfig query_pool_config{};

  const api::ContextConfig context_config{
      submit_frequency, // cmdSubmitFrequency
      cmd_config, // cmdPoolConfig
      descriptor_pool_config, // descriptorPoolConfig
      query_pool_config, // queryPoolConfig
  };

  contextConfig = context_config;

  // Empirically selected safety factor. If descriptor pools start running out
  // of memory, increase this safety factor.
  descriptorPoolSafetyFactor = 1.25;

  // For now, force kTexture3D storage as we are still developing shader support
  // for buffer storage type.
  enableStorageTypeOverride = true;
  storageTypeOverride = api::kTexture3D;

  // For now, force kWidthPacked memory layout by default as we are still
  // developing support for other memory layouts. In the future memory layout
  // settings will be serialized as part of the graph.
  enableMemoryLayoutOverride = true;
  memoryLayoutOverride = api::kWidthPacked;
}

void GraphConfig::setStorageTypeOverride(api::StorageType storage_type) {
  enableStorageTypeOverride = true;
  storageTypeOverride = storage_type;
}

void GraphConfig::setMemoryLayoutOverride(api::GPUMemoryLayout memory_layout) {
  enableMemoryLayoutOverride = true;
  memoryLayoutOverride = memory_layout;
}

} // namespace vkcompute
