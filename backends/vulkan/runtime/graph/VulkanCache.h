/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <vector>

#include <executorch/backends/vulkan/runtime/vk_api/Types.h>
#include <executorch/backends/vulkan/runtime/vk_api/memory/Buffer.h>

namespace vkcompute {

// Graph-facing view of the off-graph KV cache. The host installs one before the
// graph is built; the op function reads each layer's pool shape and wraps the
// pool buffer. Pools are [B, S, H, D], the layout the SDPA shaders index.
class VulkanCache {
 public:
  virtual ~VulkanCache() = default;

  // Pool shape for `layer`, in this backend's layout.
  virtual std::vector<int64_t> pool_sizes(int layer) const = 0;

  // Element type the pools store K/V in.
  virtual vkapi::ScalarType pool_dtype() const = 0;

  // Layers the cache was built for; the op function bounds `layer` by this.
  virtual int num_layers() const = 0;

  // The pools themselves, for the graph to wrap.
  virtual const vkapi::VulkanBuffer& k_buffer(int layer) const = 0;
  virtual const vkapi::VulkanBuffer& v_buffer(int layer) const = 0;
};

} // namespace vkcompute
