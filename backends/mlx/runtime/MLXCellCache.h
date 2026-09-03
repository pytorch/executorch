/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdexcept>
#include <vector>

#include "MLXCache.h" // AttendSpec, MLXCache
#include "MLXExecutor.h" // resolve_dtype
#include "MLXPool.h" // Pool

#include <executorch/extension/llm/cache/cache.h>
#include <executorch/extension/llm/cache/cell_cache.h>

namespace executorch {
namespace backends {
namespace mlx {

namespace cache = ::executorch::extension::llm::cache;

// The MLX byte layer behind the neutral CellCache: one Pool per layer for K and
// one for V, every pool addressed by the same cell index. update_and_fetch
// places the step, scatters the new K/V into their cells, reads the occupied
// prefix and hands over the mask the step already built.
//
// The mask is never implicit here. A cell's neighbours may belong to another
// sequence or have been freed, so a single decode token needs one as much as a
// prefill does, and neither is what MLX's fused "causal" describes.
class MLXCellCache : public cache::CellCache, public MLXCache {
 public:
  // The neutral faces come from cache::CellCache; this adds the backend one.
  void* face(cache::FaceId id) override {
    if (void* p = cache::CellCache::face(id)) {
      return p;
    }
    return cache::expose<MLXCache>(this, id);
  }

  explicit MLXCellCache(const cache::CacheConfig& cfg)
      : cache::CellCache(checked(cfg)) {
    const ::mlx::core::Dtype dt =
        resolve_dtype(static_cast<int8_t>(cfg.kv_dtype));
    kpool_.reserve(static_cast<size_t>(cfg.n_layers));
    vpool_.reserve(static_cast<size_t>(cfg.n_layers));
    for (int l = 0; l < cfg.n_layers; ++l) {
      // layers size 1 = one config broadcast to every layer, else per-layer.
      const cache::LayerConfig& lc =
          cfg.layers.size() == 1 ? cfg.layers.front() : cfg.layers[l];
      // A window bounds what a query attends, not where its token lives, so
      // every layer spans the whole cell table whatever its policy.
      kpool_.emplace_back(
          cfg.initial_capacity, cfg.capacity, lc.n_kv_heads, lc.head_dim, dt);
      vpool_.emplace_back(
          cfg.initial_capacity, cfg.capacity, lc.n_kv_heads, lc.head_dim, dt);
    }
  }

  AttendSpec update_and_fetch(
      int layer,
      const std::vector<int32_t>& positions,
      const Tensor& k,
      const Tensor& v,
      StreamOrDevice s) override {
    if (layer < 0 || layer >= static_cast<int>(kpool_.size())) {
      throw std::out_of_range("update_and_fetch: layer out of range");
    }
    // Checked before the step is placed, so a miscounted call claims no cell.
    if (static_cast<int>(positions.size()) !=
        static_cast<int>(k.shape(2))) { // BHSD: seq axis is 2
      throw std::runtime_error(
          "update_and_fetch: one position per key/value token expected");
    }
    const cache::CellStep* step = this->place_step(
        layer, positions.data(), static_cast<int>(positions.size()));
    if (!step) {
      throw std::runtime_error(
          "update_and_fetch: step undeclared, out of cells, or already served");
    }
    const size_t l = static_cast<size_t>(layer);
    kpool_[l].write_cells(step->cells, k, s);
    vpool_[l].write_cells(step->cells, v, s);
    return AttendSpec{
        kpool_[l].read(0, step->read_len, s),
        vpool_[l].read(0, step->read_len, s),
        AttendSpec::Mask::Explicit,
        mask(*step)};
  }

 private:
  // The step's bits as SDPA wants them: [1, 1, length, read_len], one row per
  // query token.
  static Tensor mask(const cache::CellStep& step) {
    return ::mlx::core::array(
        step.mask_bits.data(),
        ::mlx::core::Shape{1, 1, step.length, step.read_len},
        ::mlx::core::bool_);
  }

  // Enforce the neutral contract as an exception, the failure mode this layer
  // already uses. Runs as the base initializer's argument because CellCache's
  // own ctor indexes `layers` before this class's body does.
  static const cache::CacheConfig& checked(const cache::CacheConfig& cfg) {
    if (!cache::valid(cfg)) {
      throw std::runtime_error("MLXCellCache: invalid CacheConfig");
    }
    return cfg;
  }

  std::vector<Pool> kpool_;
  std::vector<Pool> vpool_;
};

} // namespace mlx
} // namespace backends
} // namespace executorch
