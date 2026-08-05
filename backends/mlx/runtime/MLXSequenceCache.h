/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "MLXCache.h" // AttendSpec, MLXCache
#include "MLXExecutor.h" // resolve_dtype

#include <executorch/extension/llm/cache/cache.h>
#include <executorch/extension/llm/cache/sequence_cache.h>

namespace executorch {
namespace backends {
namespace mlx {

namespace cache = ::executorch::extension::llm::cache;

// Per-layer K or V store, SDPA-major [1, H, slots, D] (cells on axis 2). The
// planner hands down physical runs (it has already applied any ring modulo), so
// the pool is layout-agnostic: flat and ring differ only in how many slots the
// layer asks for and how many runs a step produces.
class Pool {
 public:
  // initial_slots above max_slots is clamped, not rejected: the config default
  // exceeds the cap of any smaller cache, so this is the normal path.
  Pool(int initial_slots, int max_slots, int H, int D, ::mlx::core::Dtype dtype)
      : dtype_(dtype),
        max_slots_(max_slots),
        buf_(::mlx::core::zeros(
            ::mlx::core::Shape{1, H, std::min(initial_slots, max_slots), D},
            dtype)) {}

  // Place `update` at the run's physical start, casting to the storage dtype if
  // it differs.
  void write(const cache::Run& run, const Tensor& update, StreamOrDevice s) {
    const int H = static_cast<int>(buf_.shape(1));
    const int D = static_cast<int>(buf_.shape(3));
    if (run.start < 0 || run.start + run.len > max_slots_) {
      throw std::runtime_error("Pool::write: run out of bounds");
    }
    if (static_cast<int>(update.shape(2)) != run.len) {
      throw std::runtime_error("Pool::write: update length != run length");
    }
    if (static_cast<int>(update.shape(1)) != H ||
        static_cast<int>(update.shape(3)) != D) {
      throw std::runtime_error("Pool::write: K/V heads/dim mismatch");
    }
    maybe_grow(run.start + run.len, s);
    const Tensor u = update.dtype() == dtype_
        ? update
        : ::mlx::core::astype(update, dtype_, s);
    buf_ = ::mlx::core::slice_update(
        buf_,
        u,
        ::mlx::core::Shape{0, 0, run.start, 0},
        ::mlx::core::Shape{1, H, run.start + run.len, D},
        s);
  }

  // The run's cells, [start, start+len). Ring reads start mid-pool, so the run
  // start matters here as much as it does for a write.
  Tensor read(const cache::Run& run, StreamOrDevice s) const {
    const int H = static_cast<int>(buf_.shape(1));
    const int D = static_cast<int>(buf_.shape(3));
    if (run.start < 0 || run.start + run.len > slots()) {
      throw std::runtime_error("Pool::read: run out of bounds");
    }
    return ::mlx::core::slice(
        buf_,
        ::mlx::core::Shape{0, 0, run.start, 0},
        ::mlx::core::Shape{1, H, run.start + run.len, D},
        ::mlx::core::Shape{1, 1, 1, 1},
        s);
  }

  // Slots currently allocated; grows toward max_slots on demand.
  int slots() const {
    return static_cast<int>(buf_.shape(2));
  }

 private:
  // Make room for `needed` slots, growing only if the pool is short: double
  // until it fits, never past max_slots_. Cells keep their index, so growth is
  // a zero-pad on the cell axis.
  void maybe_grow(int needed, StreamOrDevice s) {
    const int cur = slots();
    if (needed <= cur) {
      return;
    }
    int next = std::max(cur, 1); // an empty pool has nothing to double
    while (next < needed) {
      next *= 2;
    }
    // The last doubling can overshoot; write() already bounds `needed` by
    // max_slots_, so clamping here cannot undershoot it.
    next = std::min(next, max_slots_);
    const int H = static_cast<int>(buf_.shape(1));
    const int D = static_cast<int>(buf_.shape(3));
    Tensor pad =
        ::mlx::core::zeros(::mlx::core::Shape{1, H, next - cur, D}, dtype_);
    buf_ = ::mlx::core::concatenate(std::vector<Tensor>{buf_, pad}, 2, s);
  }

  ::mlx::core::Dtype dtype_;
  int max_slots_;
  Tensor buf_;
};

// Bool mask [1, 1, T, S] for T queries over a span of S keys, where each query
// attends at most `window` keys ending at itself. The span is right-aligned
// (the newest key belongs to the last query), so query i spans keys
// j - i <= S - T -- the same bound MLX's "causal" applies -- and the window
// adds the lower bound j - i > S - T - window.
inline Tensor window_causal_mask(int T, int S, int window, StreamOrDevice s) {
  using namespace ::mlx::core;
  Tensor diff = subtract(
      reshape(arange(S, int32, s), Shape{1, S}, s),
      reshape(arange(T, int32, s), Shape{T, 1}, s),
      s);
  const int hi = S - T;
  Tensor band = logical_and(
      less_equal(diff, array(hi), s),
      greater_equal(diff, array(hi - window + 1), s),
      s);
  return reshape(band, Shape{1, 1, T, S}, s);
}

// The MLX byte layer behind the neutral SequenceCache: one Pool per layer for K
// and one for V, sized from that layer's own policy. update_and_fetch plans the
// step (integer runs), writes the new K/V, reads the retained window, and
// declares the mask; the op handler owns q/scale and calls SDPA.
//
// Flat and ring layers can be mixed (gemma4 alternates them): the planner emits
// physical runs either way, so they differ here only in how many slots the pool
// holds and whether a step's runs wrap.
class MLXSequenceCache : public cache::SequenceCache, public MLXCache {
 public:
  explicit MLXSequenceCache(const cache::CacheConfig& cfg)
      : cache::SequenceCache(checked(cfg)) {
    const ::mlx::core::Dtype dt =
        resolve_dtype(static_cast<int8_t>(cfg.kv_dtype));
    kpool_.reserve(static_cast<size_t>(cfg.n_layers));
    vpool_.reserve(static_cast<size_t>(cfg.n_layers));
    window_.reserve(static_cast<size_t>(cfg.n_layers));
    for (int l = 0; l < cfg.n_layers; ++l) {
      // layers size 1 = one config broadcast to every layer, else per-layer.
      const cache::LayerConfig& lc =
          cfg.layers.size() == 1 ? cfg.layers.front() : cfg.layers[l];
      const bool ring = lc.policy.kind == cache::LayerPolicy::Kind::Ring;
      window_.push_back(ring ? lc.policy.window : 0);
      // Flat retains all history, so its pool may reach the full cap and starts
      // small. A ring layer recycles a fixed window + max_write - 1 slots.
      const int max_slots = ring ? lc.policy.window +
              (cfg.max_write ? *cfg.max_write : lc.policy.window) - 1
                                 : cfg.capacity;
      const int initial = ring ? max_slots : cfg.initial_capacity;
      kpool_.emplace_back(initial, max_slots, lc.n_kv_heads, lc.head_dim, dt);
      vpool_.emplace_back(initial, max_slots, lc.n_kv_heads, lc.head_dim, dt);
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

    std::optional<cache::SeqStepPlan> p = this->plan(layer, position, T);
    if (!p) {
      throw std::runtime_error(
          "update_and_fetch: step exceeds capacity or invalid layer");
    }
    const size_t l = static_cast<size_t>(layer);
    write_runs(kpool_[l], p->write, p->n_write, k, s);
    write_runs(vpool_[l], p->write, p->n_write, v, s);
    Tensor K = read_runs(kpool_[l], p->read, p->n_read, s);
    Tensor V = read_runs(vpool_[l], p->read, p->n_read, s);
    this->commit(*p);

    // A single decode token needs no mask -- its span is exactly what it may
    // attend, on a ring layer as much as a flat one.
    if (T == 1) {
      return AttendSpec{K, V, AttendSpec::Mask::None, std::nullopt};
    }
    // A ring layer bounds each query to its own window. That only bites when
    // the window is narrower than the span (a multi-token step reads the union
    // of its queries' windows, window + T - 1 cells); otherwise plain causal is
    // exact and MLX applies it fused, with no mask tensor.
    const int S = static_cast<int>(K.shape(2));
    if (window_[l] > 0 && window_[l] < S) {
      return AttendSpec{
          K,
          V,
          AttendSpec::Mask::Explicit,
          window_causal_mask(T, S, window_[l], s)};
    }
    // MLX "causal" is lower-right aligned, so fresh and chunked prefill are
    // both correct with the new tokens at the tail.
    return AttendSpec{K, V, AttendSpec::Mask::Causal, std::nullopt};
  }

 private:
  // Scatter `update` across the step's runs. Runs are in logical order, so
  // consecutive slices of `update` map to consecutive runs. A flat step is one
  // run; a ring step splits in two when it wraps the pool.
  static void write_runs(
      Pool& pool,
      const cache::Run* runs,
      int n,
      const Tensor& update,
      StreamOrDevice s) {
    if (n == 1) {
      pool.write(runs[0], update, s);
      return;
    }
    const int H = static_cast<int>(update.shape(1));
    const int D = static_cast<int>(update.shape(3));
    int off = 0;
    for (int i = 0; i < n; ++i) {
      pool.write(
          runs[i],
          ::mlx::core::slice(
              update,
              ::mlx::core::Shape{0, 0, off, 0},
              ::mlx::core::Shape{1, H, off + runs[i].len, D},
              ::mlx::core::Shape{1, 1, 1, 1},
              s),
          s);
      off += runs[i].len;
    }
  }

  // Gather the step's runs into the retained window, oldest -> newest.
  static Tensor
  read_runs(const Pool& pool, const cache::Run* runs, int n, StreamOrDevice s) {
    if (n == 1) {
      return pool.read(runs[0], s);
    }
    std::vector<Tensor> parts;
    parts.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      parts.push_back(pool.read(runs[i], s));
    }
    return ::mlx::core::concatenate(parts, 2, s);
  }

  // Enforce the neutral contract as an exception, the failure mode this layer
  // already uses. Runs as the base initializer's argument because
  // SequenceCache's own ctor indexes `layers` before this class's body does.
  static const cache::CacheConfig& checked(const cache::CacheConfig& cfg) {
    if (!cache::valid(cfg)) {
      throw std::runtime_error("MLXSequenceCache: invalid CacheConfig");
    }
    return cfg;
  }

  std::vector<Pool> kpool_;
  std::vector<Pool> vpool_;
  std::vector<int> window_; // per layer; 0 = flat (unbounded history)
};

} // namespace mlx
} // namespace backends
} // namespace executorch
