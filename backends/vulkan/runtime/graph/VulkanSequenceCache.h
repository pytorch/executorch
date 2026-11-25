/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

#include <executorch/backends/vulkan/runtime/graph/VulkanCache.h>
#include <executorch/backends/vulkan/runtime/utils/VecUtils.h>
#include <executorch/backends/vulkan/runtime/vk_api/Exception.h>
#include <executorch/extension/llm/cache/sequence_cache.h>
#include <executorch/runtime/core/portable_type/scalar_type.h>
#include <executorch/runtime/platform/log.h>

namespace vkcompute {

namespace cache = ::executorch::extension::llm::cache;

// One sequence over flat layers. The neutral base owns the length, admission
// and rewind; this class owns one K and one V pool per layer, allocated at
// full capacity when the cache is first bound to a graph.
class VulkanSequenceCache final : public cache::SequenceCache,
                                  public VulkanCache {
 public:
  // Null if the shaders cannot serve the geometry or the storage dtype.
  static std::shared_ptr<VulkanSequenceCache> create(
      const cache::CacheGeometry& geometry,
      const cache::CacheConfig& cfg) {
    if (!cache::valid(geometry, cfg) || geometry.dflash_block_size) {
      ET_LOG(Error, "VulkanSequenceCache: invalid geometry or config");
      return nullptr;
    }
    for (const cache::LayerGeometry& layer : geometry.layers) {
      if (layer.policy.kind != cache::LayerPolicy::Kind::Flat) {
        ET_LOG(Error, "VulkanSequenceCache: only flat layers are supported");
        return nullptr;
      }
    }
    using EtScalarType = ::executorch::runtime::etensor::ScalarType;
    vkapi::ScalarType dtype;
    switch (static_cast<EtScalarType>(cfg.kv_dtype)) {
      case EtScalarType::Float:
        dtype = vkapi::kFloat;
        break;
      case EtScalarType::Half:
        dtype = vkapi::kHalf;
        break;
      default:
        ET_LOG(Error, "VulkanSequenceCache: kv_dtype must be Float or Half");
        return nullptr;
    }
    return std::shared_ptr<VulkanSequenceCache>(
        new VulkanSequenceCache(geometry, cfg, dtype));
  }

  // VulkanCache.
  void bind(vkapi::Adapter* adapter, std::shared_ptr<vkapi::Adapter> keep_alive)
      override {
    if (adapter_ != nullptr) {
      VK_CHECK_COND(
          adapter == adapter_,
          "VulkanSequenceCache: already bound to another Vulkan device");
      return;
    }
    // A device without 16-bit buffers stores half tensors as float.
    const vkapi::ScalarType stored =
        dtype_ == vkapi::kHalf && !adapter->has_full_float16_buffers_support()
        ? vkapi::kFloat
        : dtype_;
    std::vector<vkapi::VulkanBuffer> k_pools, v_pools;
    k_pools.reserve(layers_.size());
    v_pools.reserve(layers_.size());
    for (size_t layer = 0; layer < layers_.size(); ++layer) {
      const cache::LayerGeometry& lg = layers_[layer];
      // Buffer tensors pad the packed (head) dim to a multiple of 4.
      const int64_t nbytes = int64_t{capacity()} * lg.n_kv_heads *
          utils::align_up_4(lg.head_dim) * vkapi::element_size(stored);
      VK_CHECK_COND(
          nbytes <= adapter->max_buffer_numel(),
          "VulkanSequenceCache: layer ",
          layer,
          " needs a ",
          nbytes,
          "-byte pool, over the device's storage buffer limit");
      k_pools.push_back(adapter->vma().create_storage_buffer(nbytes));
      v_pools.push_back(adapter->vma().create_storage_buffer(nbytes));
    }
    keep_alive_ = std::move(keep_alive);
    k_pools_ = std::move(k_pools);
    v_pools_ = std::move(v_pools);
    adapter_ = adapter;
  }

  int num_layers() const override {
    return static_cast<int>(layers_.size());
  }

  vkapi::ScalarType pool_dtype() const override {
    return dtype_;
  }

  std::vector<int64_t> pool_sizes(int layer) const override {
    const cache::LayerGeometry& lg = layers_.at(layer);
    return {1, capacity(), lg.n_kv_heads, lg.head_dim};
  }

  const vkapi::VulkanBuffer& k_buffer(int layer) const override {
    return k_pools_.at(layer);
  }

  const vkapi::VulkanBuffer& v_buffer(int layer) const override {
    return v_pools_.at(layer);
  }

  // Each layer advances once per forward, in any order, to the same end, and
  // the shared length is the furthest any layer reached.
  bool admit(int layer, int position, int n) override {
    if (position < 0 || n <= 0 || n > capacity() || layer < 0 ||
        layer >= num_layers() || position != layer_lengths_[layer]) {
      return false;
    }
    const std::optional<cache::SeqStepPlan> step = plan(layer, position, n);
    if (!step) {
      return false;
    }
    commit(*step);
    layer_lengths_[layer] = position + n;
    return true;
  }

  // SequenceControl.
  bool rewind(int position) override {
    if (!cache::SequenceCache::rewind(position)) {
      return false;
    }
    for (int& layer_length : layer_lengths_) {
      layer_length = std::min(layer_length, position);
    }
    return true;
  }

  void clear() override {
    cache::SequenceCache::clear();
    std::fill(layer_lengths_.begin(), layer_lengths_.end(), 0);
  }

 protected:
  void* face(cache::FaceId id) override {
    if (void* p = cache::SequenceCache::face(id)) {
      return p;
    }
    return cache::expose<VulkanCache>(this, id);
  }

 private:
  VulkanSequenceCache(
      const cache::CacheGeometry& geometry,
      const cache::CacheConfig& cfg,
      const vkapi::ScalarType dtype)
      : cache::SequenceCache(geometry, cfg),
        layers_(geometry.layers),
        dtype_(dtype),
        layer_lengths_(geometry.layers.size(), 0) {}

  std::vector<cache::LayerGeometry> layers_;
  vkapi::ScalarType dtype_;
  // How far each layer's pools have been written.
  std::vector<int> layer_lengths_;
  vkapi::Adapter* adapter_ = nullptr;
  // Declared before the pools so it is released after them.
  std::shared_ptr<vkapi::Adapter> keep_alive_;
  std::vector<vkapi::VulkanBuffer> k_pools_;
  std::vector<vkapi::VulkanBuffer> v_pools_;
};

} // namespace vkcompute
