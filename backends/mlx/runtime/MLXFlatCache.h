/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "MLXCacheImpl.h" // AttendSpec, MLXCacheImpl
#include "MLXExecutor.h" // to_shape

#include <executorch/extension/llm/cache/cache.h>
#include <executorch/extension/llm/cache/sequence_cache.h>

namespace executorch {
namespace backends {
namespace mlx {

namespace cache = ::executorch::extension::llm::cache;

// Per-layer K or V store, SDPA-major [1, H, capacity, D] (cells on axis 2),
// allocated at construction from config H/D/dtype. A write is a slice_update
// run (casting the update to the storage dtype if it differs); a read is a [0,
// len) slice.
class FlatPool {
 public:
  FlatPool(int capacity, int H, int D, ::mlx::core::Dtype dtype)
      : dtype_(dtype),
        buf_(::mlx::core::zeros(
            to_shape(std::vector<int>{1, H, capacity, D}),
            dtype)) {}

  void write(int start, const Tensor& update, StreamOrDevice s) {
    const int cap = static_cast<int>(buf_.shape(2));
    const int H = static_cast<int>(buf_.shape(1));
    const int D = static_cast<int>(buf_.shape(3));
    const int T = static_cast<int>(update.shape(2));
    if (start < 0 || start + T > cap) {
      throw std::runtime_error("FlatPool::write: run out of bounds");
    }
    if (static_cast<int>(update.shape(1)) != H ||
        static_cast<int>(update.shape(3)) != D) {
      throw std::runtime_error("FlatPool::write: K/V heads/dim mismatch");
    }
    const Tensor u = update.dtype() == dtype_
        ? update
        : ::mlx::core::astype(update, dtype_, s);
    std::vector<int> lo{0, 0, start, 0};
    std::vector<int> hi{1, H, start + T, D};
    buf_ = ::mlx::core::slice_update(buf_, u, to_shape(lo), to_shape(hi), s);
  }

  Tensor read(int len, StreamOrDevice s) const {
    const int H = static_cast<int>(buf_.shape(1));
    const int D = static_cast<int>(buf_.shape(3));
    std::vector<int> lo{0, 0, 0, 0};
    std::vector<int> hi{1, H, len, D};
    std::vector<int> step{1, 1, 1, 1};
    return ::mlx::core::slice(
        buf_, to_shape(lo), to_shape(hi), to_shape(step), s);
  }

 private:
  ::mlx::core::Dtype dtype_;
  Tensor buf_;
};

// Single-sequence, full-history cache: the neutral SequenceCache bookkeeping
// over per-layer FlatPools. update_and_fetch plans the step (integer runs),
// writes the new K/V, reads the retained window, and declares the mask; the op
// handler owns q/scale and calls SDPA.
class MLXFlatCache : public cache::SequenceCache, public MLXCacheImpl {
 public:
  explicit MLXFlatCache(const cache::CacheConfig& cfg)
      : cache::SequenceCache(cfg) {
    if (!cfg.kv_dtype) {
      throw std::runtime_error(
          "MLXFlatCache: CacheConfig::kv_dtype (storage precision) must be set");
    }
    const ::mlx::core::Dtype dt =
        resolve_dtype(static_cast<int8_t>(*cfg.kv_dtype));
    kpool_.reserve(static_cast<size_t>(cfg.n_layers));
    vpool_.reserve(static_cast<size_t>(cfg.n_layers));
    for (int l = 0; l < cfg.n_layers; ++l) {
      // layers size 1 = one config broadcast to every layer, else per-layer.
      const cache::LayerConfig& lc =
          cfg.layers.size() == 1 ? cfg.layers.front() : cfg.layers[l];
      kpool_.emplace_back(cfg.capacity, lc.n_kv_heads, lc.head_dim, dt);
      vpool_.emplace_back(cfg.capacity, lc.n_kv_heads, lc.head_dim, dt);
    }
  }

  AttendSpec update_and_fetch(
      int layer,
      int position,
      const Tensor& k,
      const Tensor& v,
      StreamOrDevice s) override {
    if (layer < 0 || layer >= static_cast<int>(kpool_.size())) {
      throw std::out_of_range("update_and_fetch: layer out of range");
    }
    const int T = static_cast<int>(k.shape(2)); // BHSD: seq axis is 2
    const int start = position;

    std::optional<cache::SeqStepPlan> p = this->plan(layer, start, T);
    if (!p) {
      throw std::runtime_error(
          "update_and_fetch: step exceeds capacity or invalid layer");
    }
    // Flat is single-run: one write [start, start+T), one read [0, end). A
    // wrapping (two-run) plan means a ring layer, which the ring cache handles;
    // reject it here rather than silently drop the second run.
    if (p->n_write != 1 || p->n_read != 1) {
      throw std::runtime_error(
          "update_and_fetch: expected a flat single-run plan");
    }
    const size_t l = static_cast<size_t>(layer);
    kpool_[l].write(p->write[0].start, k, s);
    vpool_[l].write(p->write[0].start, v, s);
    Tensor K = kpool_[l].read(p->read[0].len, s);
    Tensor V = vpool_[l].read(p->read[0].len, s);
    this->commit(*p);

    // A multi-token chain is Causal (MLX "causal" is lower-right aligned, so
    // fresh and chunked prefill are both correct with new tokens at the tail);
    // a single decode token needs no mask.
    const AttendSpec::Mask kind =
        (T > 1) ? AttendSpec::Mask::Causal : AttendSpec::Mask::None;
    return AttendSpec{K, V, kind, std::nullopt};
  }

 private:
  std::vector<FlatPool> kpool_;
  std::vector<FlatPool> vpool_;
};

} // namespace mlx
} // namespace backends
} // namespace executorch
