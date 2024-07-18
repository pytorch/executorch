/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>

namespace vkcompute {

struct GraphConfig final {
  api::ContextConfig context_config;

  // Creating a descriptor pool with exactly the number of descriptors tallied
  // by iterating through the shader layouts of shaders used in the graph risks
  // the descriptor pool running out of memory, therefore apply a safety factor
  // to descriptor counts when creating the descriptor pool to mitigate this
  // risk.
  float descriptor_pool_safety_factor;

  bool enable_storage_type_override;
  utils::StorageType storage_type_override;

  bool enable_memory_layout_override;
  utils::GPUMemoryLayout memory_layout_override;

  bool enable_querypool;

  bool enable_local_wg_size_override;
  utils::uvec3 local_wg_size_override;

  // Generate a default graph config with pre-configured settings
  explicit GraphConfig();

  void set_storage_type_override(utils::StorageType storage_type);
  void set_memory_layout_override(utils::GPUMemoryLayout memory_layout);
  void set_local_wg_size_override(const utils::uvec3& local_wg_size);
};

} // namespace vkcompute
