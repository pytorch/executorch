/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// The byte layer behind the neutral SequenceCache: host buffers written at the
// slots the planner names, handed back as raw pointers. Flat layers only; a
// ring step can produce two runs, which this rejects.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

#include <executorch/backends/xnnpack/runtime/XnnCache.h>
#include <executorch/extension/llm/cache/cache.h>
#include <executorch/extension/llm/cache/sequence_cache.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/result.h>

namespace executorch {
namespace backends {
namespace xnnpack {

namespace cache = ::executorch::extension::llm::cache;

// Per-layer K or V store, BHSD-major [1, H, slots, D] -- the layout the
// attention kernel reads directly. Cells keep their index as the pool grows,
// but the row stride widens with it, so growth moves every head.
class Pool {
 public:
  // initial_slots above max_slots is clamped: the config default exceeds the
  // cap of any smaller cache, so this is the normal path.
  Pool(int initial_slots, int max_slots, int H, int D, size_t elem_size)
      : h_(H),
        d_(D),
        elem_size_(elem_size),
        max_slots_(max_slots),
        slots_(std::min(initial_slots, max_slots)),
        buf_(bytes(slots_), 0) {}

  // Copy run.len tokens, from token `src_tok` of a dense [1, H, n_tok, D]
  // source, into slots [run.start, run.start + run.len). Per head: the source's
  // row stride is n_tok, the pool's is slots.
  runtime::Error
  write(const cache::Run& run, const void* src, int n_tok, int src_tok) {
    ET_CHECK_OR_RETURN_ERROR(
        run.start >= 0 && run.len >= 0 && run.start + run.len <= max_slots_,
        InvalidArgument,
        "Pool::write: run [%d, %d) out of bounds for %d slots",
        run.start,
        run.start + run.len,
        max_slots_);
    ET_CHECK_OR_RETURN_ERROR(
        src_tok >= 0 && run.len <= n_tok - src_tok,
        InvalidArgument,
        "Pool::write: source tokens [%d, %d) out of bounds for %d tokens",
        src_tok,
        src_tok + run.len,
        n_tok);
    ET_CHECK_OK_OR_RETURN_ERROR(maybe_grow(run.start + run.len));

    const auto* s = static_cast<const uint8_t*>(src);
    const size_t row = static_cast<size_t>(d_) * elem_size_;
    for (int h = 0; h < h_; ++h) {
      std::memcpy(
          buf_.data() + (static_cast<size_t>(h) * slots_ + run.start) * row,
          s + (static_cast<size_t>(h) * n_tok + src_tok) * row,
          static_cast<size_t>(run.len) * row);
    }
    return runtime::Error::Ok;
  }

  const void* data() const {
    return buf_.data();
  }

  // Slots currently allocated; grows toward max_slots on demand.
  int slots() const {
    return slots_;
  }

 private:
  // Double until `needed` fits, never past max_slots_. Widening the row stride
  // moves every head's cells.
  runtime::Error maybe_grow(int needed) {
    if (needed <= slots_) {
      return runtime::Error::Ok;
    }
    int next = std::max(slots_, 1); // an empty pool has nothing to double
    while (next < needed) {
      next *= 2;
    }
    // The last doubling can overshoot; write() already bounds `needed` by
    // max_slots_, so clamping here cannot undershoot it.
    next = std::min(next, max_slots_);

    std::vector<uint8_t> grown(bytes(next), 0);
    const size_t row = static_cast<size_t>(d_) * elem_size_;
    if (slots_ > 0) { // an empty pool has no buffer to read from
      for (int h = 0; h < h_; ++h) {
        std::memcpy(
            grown.data() + static_cast<size_t>(h) * next * row,
            buf_.data() + static_cast<size_t>(h) * slots_ * row,
            static_cast<size_t>(slots_) * row);
      }
    }
    buf_.swap(grown);
    slots_ = next;
    return runtime::Error::Ok;
  }

  size_t bytes(int slots) const {
    return static_cast<size_t>(h_) * slots * d_ * elem_size_;
  }

  int h_;
  int d_;
  size_t elem_size_;
  int max_slots_;
  int slots_;
  std::vector<uint8_t> buf_;
};

// One Pool per layer for K and one for V, sized from that layer's config.
// update_and_fetch plans the step, writes the new K/V where the plan says, and
// hands back the window; attending over it is the caller's half.
class XnnSequenceCache : public cache::SequenceCache, public XnnCache {
 public:
  // Validates the config, which a constructor cannot do without exceptions.
  static runtime::Result<std::unique_ptr<XnnSequenceCache>> create(
      const cache::CacheConfig& cfg) {
    ET_CHECK_OR_RETURN_ERROR(
        cache::valid(cfg), InvalidArgument, "XnnSequenceCache: invalid config");
    for (const cache::LayerConfig& lc : cfg.layers) {
      ET_CHECK_OR_RETURN_ERROR(
          lc.policy.kind == cache::LayerPolicy::Kind::Flat,
          NotSupported,
          "XnnSequenceCache: only flat layers are supported");
      ET_CHECK_OR_RETURN_ERROR(
          lc.n_kv_heads > 0 && lc.head_dim > 0,
          InvalidArgument,
          "XnnSequenceCache: n_kv_heads and head_dim must be positive");
    }
    // Fixed-width float elements are all the pool stores. A quantized storage
    // dtype would also need scales and zero points, which it does not carry.
    const auto dtype = static_cast<aten::ScalarType>(cfg.kv_dtype);
    ET_CHECK_OR_RETURN_ERROR(
        dtype == aten::ScalarType::Float || dtype == aten::ScalarType::Half ||
            dtype == aten::ScalarType::BFloat16,
        NotSupported,
        "XnnSequenceCache: unsupported kv_dtype %d",
        cfg.kv_dtype);
    return std::unique_ptr<XnnSequenceCache>(
        new XnnSequenceCache(cfg, runtime::elementSize(dtype)));
  }

  runtime::Result<AttendSpec> update_and_fetch(
      int layer,
      int position,
      const void* k,
      const void* v,
      int n_tok) override {
    ET_CHECK_OR_RETURN_ERROR(
        layer >= 0 && layer < static_cast<int>(kpool_.size()),
        InvalidArgument,
        "update_and_fetch: layer %d out of range",
        layer);

    std::optional<cache::SeqStepPlan> p = this->plan(layer, position, n_tok);
    ET_CHECK_OR_RETURN_ERROR(
        p.has_value(),
        InvalidArgument,
        "update_and_fetch: step at %d + %d exceeds capacity",
        position,
        n_tok);
    // `valid_len` below is a length measured from slot 0, so the window has to
    // be the pool's prefix and the step's tokens have to land in one run. A
    // ring layer breaks both -- and breaks the prefix even when its read does
    // not wrap, since that single run still starts mid-pool.
    ET_CHECK_OR_RETURN_ERROR(
        p->n_write == 1 && p->write[0].len == n_tok && p->n_read == 1 &&
            p->read[0].start == 0,
        NotSupported,
        "update_and_fetch: only a single-run step reading from slot 0 is supported");

    const auto l = static_cast<size_t>(layer);
    ET_CHECK_OK_OR_RETURN_ERROR(kpool_[l].write(p->write[0], k, n_tok, 0));
    ET_CHECK_OK_OR_RETURN_ERROR(vpool_[l].write(p->write[0], v, n_tok, 0));
    this->commit(*p);

    // One query attends its whole window; a multi-token step is causal,
    // right-aligned because writes append. Flat never needs Explicit.
    return AttendSpec{
        kpool_[l].data(),
        vpool_[l].data(),
        kpool_[l].slots(),
        p->read[0].len,
        n_tok == 1 ? AttendSpec::Mask::None : AttendSpec::Mask::Causal,
        nullptr};
  }

 private:
  XnnSequenceCache(const cache::CacheConfig& cfg, size_t elem_size)
      : cache::SequenceCache(cfg) {
    kpool_.reserve(static_cast<size_t>(cfg.n_layers));
    vpool_.reserve(static_cast<size_t>(cfg.n_layers));
    for (int l = 0; l < cfg.n_layers; ++l) {
      // layers size 1 = one config broadcast to every layer, else per-layer.
      const cache::LayerConfig& lc =
          cfg.layers.size() == 1 ? cfg.layers.front() : cfg.layers[l];
      // Flat retains all history, so its pool may reach the full capacity.
      kpool_.emplace_back(
          cfg.initial_capacity,
          cfg.capacity,
          lc.n_kv_heads,
          lc.head_dim,
          elem_size);
      vpool_.emplace_back(
          cfg.initial_capacity,
          cfg.capacity,
          lc.n_kv_heads,
          lc.head_dim,
          elem_size);
    }
  }

  std::vector<Pool> kpool_;
  std::vector<Pool> vpool_;
};

} // namespace xnnpack
} // namespace backends
} // namespace executorch
