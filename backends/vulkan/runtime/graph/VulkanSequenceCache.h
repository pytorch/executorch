/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// The Vulkan byte layer behind the neutral SequenceCache. Holds one buffer
// pool per layer for K and one for V, allocated at construction.
//
// Flat layers only: every dispatch here assumes slot == position, which a ring
// layer breaks.

#include <memory>
#include <vector>

#include <executorch/backends/vulkan/runtime/api/containers/Tensor.h>
#include <executorch/backends/vulkan/runtime/graph/VulkanCache.h>
#include <executorch/extension/llm/cache/cache.h>
#include <executorch/extension/llm/cache/sequence_cache.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/portable_type/scalar_type.h>
#include <executorch/runtime/core/result.h>

namespace vkcompute {

namespace cache = ::executorch::extension::llm::cache;
namespace runtime = ::executorch::runtime;

class VulkanSequenceCache : public cache::SequenceCache, public VulkanCache {
 public:
  static runtime::Result<std::unique_ptr<VulkanSequenceCache>> create(
      const cache::CacheConfig& cfg) {
    ET_CHECK_OR_RETURN_ERROR(
        cache::valid(cfg),
        InvalidArgument,
        "VulkanSequenceCache: invalid config");
    for (const cache::LayerConfig& lc : cfg.layers) {
      ET_CHECK_OR_RETURN_ERROR(
          lc.policy.kind == cache::LayerPolicy::Kind::Flat,
          NotSupported,
          "VulkanSequenceCache: only flat layers are supported");
      ET_CHECK_OR_RETURN_ERROR(
          lc.n_kv_heads > 0 && lc.head_dim > 0,
          InvalidArgument,
          "VulkanSequenceCache: n_kv_heads and head_dim must be positive");
    }
    // The SDPA shaders are only generated for fp32 and fp16.
    using EtScalarType = ::executorch::runtime::etensor::ScalarType;
    vkapi::ScalarType pool_dtype;
    switch (static_cast<EtScalarType>(cfg.kv_dtype)) {
      case EtScalarType::Float:
        pool_dtype = vkapi::kFloat;
        break;
      case EtScalarType::Half:
        pool_dtype = vkapi::kHalf;
        break;
      default:
        ET_LOG(Error, "VulkanSequenceCache: unsupported kv_dtype");
        return runtime::Error::NotSupported;
    }
    return std::unique_ptr<VulkanSequenceCache>(
        new VulkanSequenceCache(cfg, pool_dtype));
  }

  // [B, S, H, D], the layout the SDPA shaders index. S is the allocated depth.
  std::vector<int64_t> pool_sizes(int layer) const override {
    const cache::LayerConfig& lc = layers_[static_cast<size_t>(layer)];
    return {1, capacity(), lc.n_kv_heads, lc.head_dim};
  }

  vkapi::ScalarType pool_dtype() const override {
    return pool_dtype_;
  }

  int num_layers() const override {
    return static_cast<int>(layers_.size());
  }

  const vkapi::VulkanBuffer& k_buffer(int layer) const override {
    return kpool_[static_cast<size_t>(layer)].buffer();
  }
  const vkapi::VulkanBuffer& v_buffer(int layer) const override {
    return vpool_[static_cast<size_t>(layer)].buffer();
  }

 private:
  VulkanSequenceCache(
      const cache::CacheConfig& cfg,
      const vkapi::ScalarType pool_dtype)
      : cache::SequenceCache(cfg), pool_dtype_(pool_dtype) {
    layers_.reserve(static_cast<size_t>(cfg.n_layers));
    for (int l = 0; l < cfg.n_layers; ++l) {
      // layers size 1 = one config broadcast to every layer, else per-layer.
      layers_.push_back(
          cfg.layers.size() == 1 ? cfg.layers.front() : cfg.layers[l]);
    }
    // The global context outlives every graph.
    kpool_.reserve(static_cast<size_t>(cfg.n_layers));
    vpool_.reserve(static_cast<size_t>(cfg.n_layers));
    for (int l = 0; l < cfg.n_layers; ++l) {
      for (auto* pool : {&kpool_, &vpool_}) {
        pool->emplace_back(
            api::context(),
            pool_sizes(l),
            pool_dtype_,
            utils::kBuffer,
            utils::kWidthPacked);
      }
    }
  }

  std::vector<cache::LayerConfig> layers_;
  vkapi::ScalarType pool_dtype_;
  std::vector<api::vTensor> kpool_;
  std::vector<api::vTensor> vpool_;
};

} // namespace vkcompute
