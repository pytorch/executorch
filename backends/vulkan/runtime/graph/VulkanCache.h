/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <executorch/backends/vulkan/runtime/vk_api/Adapter.h>
#include <executorch/backends/vulkan/runtime/vk_api/Types.h>
#include <executorch/backends/vulkan/runtime/vk_api/memory/Buffer.h>

namespace vkcompute {

// Graph-facing face of the off-graph KV cache. Each layer has one K and one V
// pool, a [1, capacity, H_kv, D] buffer in the layout the LLM-mode SDPA shaders
// index. update_and_attend wraps the pools as tensors and admits each step.
class VulkanCache {
 public:
  static constexpr const char* kFaceName = "vulkan.VulkanCache";

  virtual ~VulkanCache() = default;

  // Allocates the pools on `adapter` on the first call; every later call must
  // pass the same adapter. `keep_alive` owns the adapter unless it is the
  // process-global one, and is held for as long as the pools are.
  virtual void bind(
      vkapi::Adapter* adapter,
      std::shared_ptr<vkapi::Adapter> keep_alive) = 0;

  virtual int num_layers() const = 0;
  virtual vkapi::ScalarType pool_dtype() const = 0;
  virtual std::vector<int64_t> pool_sizes(int layer) const = 0;
  virtual const vkapi::VulkanBuffer& k_buffer(int layer) const = 0;
  virtual const vkapi::VulkanBuffer& v_buffer(int layer) const = 0;

  // Admits `n` tokens at consecutive positions from `position` into `layer`
  // and advances the cache length past them, before the graph runs them, so a
  // runner rewinds after a failed forward. A step is admitted if it fits and
  // starts where that layer's history ends, so each layer admits a step once
  // and running it again takes a rewind. False = not admitted, and nothing
  // changed.
  virtual bool admit(int layer, int position, int n) = 0;
};

} // namespace vkcompute
