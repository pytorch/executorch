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
  api::ContextConfig contextConfig;

  // Creating a descriptor pool with exactly the number of descriptors tallied
  // by iterating through the shader layouts of shaders used in the graph risks
  // the descriptor pool running out of memory, therefore apply a safety factor
  // to descriptor counts when creating the descriptor pool to mitigate this
  // risk.
  float descriptorPoolSafetyFactor;

  bool enableStorageTypeOverride;
  api::StorageType storageTypeOverride;

  bool enableMemoryLayoutOverride;
  api::GPUMemoryLayout memoryLayoutOverride;

  // Generate a default graph config with pre-configured settings
  explicit GraphConfig();

  void setStorageTypeOverride(api::StorageType storage_type);
  void setMemoryLayoutOverride(api::GPUMemoryLayout memory_layout);
};

} // namespace vkcompute
