/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <map>
#include <stdexcept>
#include <vector>

#include "MLXCache.h" // AttendSpec, MLXCache
#include "MLXExecutor.h" // resolve_dtype
#include "MLXPool.h" // Pool
#include "MLXSequenceCache.h" // read_runs, write_runs, window_causal_mask

#include <executorch/extension/llm/cache/batched_sequence_cache.h>
#include <executorch/extension/llm/cache/cache.h>

namespace executorch {
namespace backends {
namespace mlx {

namespace cache = ::executorch::extension::llm::cache;

// The MLX byte layer behind the neutral BatchedSequenceCache: one pool pair per
// layer per sequence, so a step's spans write and read entirely within their
// own sequence. update_and_fetch places the step, then answers each span from
// that sequence's pools.
//
// Nothing here builds a mask. A span sees only its own sequence's cells, which
// is what MLX's fused forms already describe -- no mask for a decode token,
// "causal" for a prefill -- and only a window narrower than the span needs one,
// exactly as it does for a single sequence.
class MLXBatchedSequenceCache : public cache::BatchedSequenceCache,
                                public MLXCache {
 public:
  explicit MLXBatchedSequenceCache(const cache::CacheConfig& cfg)
      : cache::BatchedSequenceCache(checked(cfg)), cfg_(cfg) {}

  Tensor attend(
      int layer,
      const std::vector<int32_t>& positions,
      const Tensor& q,
      const Tensor& k,
      const Tensor& v,
      float scale,
      StreamOrDevice s) override {
    if (layer < 0 || layer >= cfg_.n_layers) {
      throw std::out_of_range("attend: layer out of range");
    }
    // Checked before the step is placed, so a miscounted call writes nothing.
    if (static_cast<int>(positions.size()) !=
        static_cast<int>(k.shape(2))) { // BHSD: seq axis is 2
      throw std::runtime_error(
          "attend: one position per key/value token expected");
    }
    const std::vector<cache::SeqSpan>* spans = this->place_step(
        layer, positions.data(), static_cast<int>(positions.size()));
    if (!spans) {
      throw std::runtime_error(
          "attend: step undeclared, out of room, out of order, "
          "or already served");
    }

    // Each span reads only its own sequence, so the spans attend
    // independently and rejoin in the order the step declared them.
    std::vector<Tensor> outs;
    outs.reserve(spans->size());
    const size_t l = static_cast<size_t>(layer);
    int at = 0;
    for (const cache::SeqSpan& span : *spans) {
      LayerPools& row = row_for(span.seq_id);
      write_runs(
          row.kpool[l],
          span.plan.write,
          span.plan.n_write,
          tokens(k, at, span.q_len, s),
          s);
      write_runs(
          row.vpool[l],
          span.plan.write,
          span.plan.n_write,
          tokens(v, at, span.q_len, s),
          s);
      const AttendSpec sp = spec_for(
          read_runs(row.kpool[l], span.plan.read, span.plan.n_read, s),
          read_runs(row.vpool[l], span.plan.read, span.plan.n_read, s),
          span.q_len,
          row.window[l],
          s);
      outs.push_back(::executorch::backends::mlx::attend(
          sp, tokens(q, at, span.q_len, s), scale, s));
      at += span.q_len;
    }
    return outs.size() == 1 ? std::move(outs.front())
                            : ::mlx::core::concatenate(outs, 2, s);
  }

  // Copies the donor's pools whole, not just its first `upto` cells: a ring
  // addresses slots by position, so the same pools at the length the base set
  // hold the same history.
  std::optional<int32_t> seq_clone(int32_t src, std::optional<int> upto)
      override {
    const std::optional<int32_t> dst =
        cache::BatchedSequenceCache::seq_clone(src, upto);
    if (!dst) {
      return std::nullopt;
    }
    auto it = rows_.find(src);
    if (it != rows_.end()) {
      rows_.emplace(*dst, it->second);
    }
    return dst;
  }

  // The pools go with the sequence; the base frees the bookkeeping.
  bool seq_rm(int32_t seq_id) override {
    if (!cache::BatchedSequenceCache::seq_rm(seq_id)) {
      return false;
    }
    rows_.erase(seq_id);
    return true;
  }

  void clear() override {
    cache::BatchedSequenceCache::clear();
    rows_.clear();
  }

 protected:
  void* face(cache::FaceId id) override {
    if (void* p = cache::BatchedSequenceCache::face(id)) {
      return p;
    }
    return cache::expose<MLXCache>(this, id);
  }

 private:
  // The step's tokens for one span, on the sequence axis.
  static Tensor tokens(const Tensor& t, int off, int len, StreamOrDevice s) {
    if (off == 0 && len == static_cast<int>(t.shape(2))) {
      return t;
    }
    return ::mlx::core::slice(
        t,
        ::mlx::core::Shape{0, 0, off, 0},
        ::mlx::core::Shape{t.shape(0), t.shape(1), off + len, t.shape(3)},
        s);
  }

  // Pools are per sequence, so they are built when one first writes rather than
  // reserved for every id the cache could hand out. initial_capacity is paid
  // once per sequence per layer, not once per layer.
  LayerPools& row_for(int32_t seq_id) {
    auto it = rows_.find(seq_id);
    if (it != rows_.end()) {
      return it->second;
    }
    return rows_.emplace(seq_id, make_pools(cfg_)).first->second;
  }

  static const cache::CacheConfig& checked(const cache::CacheConfig& cfg) {
    if (!cache::valid(cfg)) {
      throw std::runtime_error("MLXBatchedSequenceCache: invalid CacheConfig");
    }
    return cfg;
  }

  cache::CacheConfig cfg_;
  std::map<int32_t, LayerPools> rows_;
};

} // namespace mlx
} // namespace backends
} // namespace executorch
