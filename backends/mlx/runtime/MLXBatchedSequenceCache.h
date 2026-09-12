/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

#include "MLXCache.h"
#include "MLXSequenceCache.h"

#include <executorch/extension/llm/cache/cache.h>

namespace executorch {
namespace backends {
namespace mlx {

namespace cache = ::executorch::extension::llm::cache;

// A private MLXSequenceCache per sequence, with one shared logical capacity.
// A flat token axis is split into consecutive spans and each span attends only
// over its sequence's history before the results are joined in token order.
class MLXBatchedSequenceCache : public cache::Cache,
                                public cache::BatchControl,
                                public MLXCache {
 public:
  explicit MLXBatchedSequenceCache(const cache::CacheConfig& cfg)
      : cfg_(checked(cfg)) {}

  int capacity() const override {
    return cfg_.capacity;
  }

  void clear() override {
    rows_.clear();
    invalidate_step();
  }

  bool declare_step(const std::vector<int32_t>& seq_ids) override {
    if (seq_ids.empty() || !has_room(static_cast<int>(seq_ids.size()))) {
      return false;
    }
    for (int32_t id : seq_ids) {
      if (rows_.find(id) == rows_.end()) {
        return false;
      }
    }

    invalidate_step();
    for (size_t i = 0; i < seq_ids.size();) {
      size_t j = i + 1;
      while (j < seq_ids.size() && seq_ids[j] == seq_ids[i]) {
        ++j;
      }
      spans_.push_back(Span{seq_ids[i], static_cast<int>(j - i)});
      i = j;
    }
    declared_ = true;
    return true;
  }

  std::optional<int> max_seqs() const override {
    return std::nullopt;
  }

  std::optional<int32_t> seq_new() override {
    const int32_t id = free_id();
    rows_.try_emplace(id, cfg_);
    return id;
  }

  std::optional<int32_t> seq_clone(int32_t src, std::optional<int> upto)
      override {
    auto it = rows_.find(src);
    if (it == rows_.end() || it->second.length() == 0) {
      return std::nullopt;
    }

    const int keep =
        upto ? std::min(*upto, it->second.length()) : it->second.length();
    // Checked here so the constructor is only given a position it can honour.
    if (keep <= 0 || !has_room(keep) || !it->second.can_rewind(keep)) {
      return std::nullopt;
    }

    // Forking copies cells, and an unbound cache has no stream to run that on.
    if (!controller_) {
      return std::nullopt;
    }
    const int32_t dst = free_id();
    rows_.try_emplace(dst, it->second, keep, *controller_);
    invalidate_step();
    return dst;
  }

  bool seq_rm(int32_t seq_id) override {
    if (rows_.erase(seq_id) == 0) {
      return false;
    }
    invalidate_step();
    return true;
  }

  bool rewind(int32_t seq_id, int position) override {
    auto it = rows_.find(seq_id);
    if (it == rows_.end() || position < 0 || !it->second.rewind(position)) {
      return false;
    }
    invalidate_step();
    return true;
  }

  int pos(int32_t seq_id) const override {
    auto it = rows_.find(seq_id);
    return it == rows_.end() ? 0 : it->second.length();
  }

  Tensor attend(
      int layer,
      const std::vector<int32_t>& positions,
      const Tensor& q,
      const Tensor& k,
      const Tensor& v,
      float scale,
      StreamOrDevice s) override {
    validate_step(layer, positions, q, k, v);

    if (!placed_) {
      step_pos_ = positions;
      declared_ = false;
      placed_ = true;
    }

    std::vector<Tensor> outs;
    outs.reserve(spans_.size());
    int at = 0;
    for (const Span& span : spans_) {
      const std::vector<int32_t> span_positions(
          positions.begin() + at, positions.begin() + at + span.q_len);
      outs.push_back(rows_.at(span.seq_id)
                         .attend(
                             layer,
                             span_positions,
                             tokens(q, at, span.q_len, s),
                             tokens(k, at, span.q_len, s),
                             tokens(v, at, span.q_len, s),
                             scale,
                             s));
      at += span.q_len;
    }
    return outs.size() == 1 ? std::move(outs.front())
                            : ::mlx::core::concatenate(outs, 2, s);
  }

  void bind_controller_stream(::mlx::core::Stream s) override {
    controller_ = s;
  }

 protected:
  void* face(cache::FaceId id) override {
    return cache::expose<cache::BatchControl, MLXCache>(this, id);
  }

 private:
  struct Span {
    int32_t seq_id;
    int q_len;
  };

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

  void validate_step(
      int layer,
      const std::vector<int32_t>& positions,
      const Tensor& q,
      const Tensor& k,
      const Tensor& v) const {
    if (layer < 0 || layer >= cfg_.n_layers) {
      throw std::out_of_range("attend: layer out of range");
    }
    if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
      throw std::runtime_error("attend: Q/K/V must be BHSD tensors");
    }
    const int length = static_cast<int>(positions.size());
    if (static_cast<int>(q.shape(2)) != length ||
        static_cast<int>(k.shape(2)) != length ||
        static_cast<int>(v.shape(2)) != length) {
      throw std::runtime_error(
          "attend: one position per query/key/value token expected");
    }
    if (placed_) {
      if (positions != step_pos_) {
        throw std::runtime_error("attend: positions differ from placed step");
      }
    } else if (
        !declared_ || length != step_length() || !check_positions(positions)) {
      throw std::runtime_error(
          "attend: step undeclared, out of room, or out of order");
    }

    int at = 0;
    for (const Span& span : spans_) {
      if (!rows_.at(span.seq_id).plan(layer, positions[at], span.q_len)) {
        throw std::runtime_error("attend: step exceeds sequence capacity");
      }
      at += span.q_len;
    }
  }

  bool check_positions(const std::vector<int32_t>& positions) const {
    std::map<int32_t, int> ends;
    int at = 0;
    for (const Span& span : spans_) {
      auto end = ends.find(span.seq_id);
      int want =
          end == ends.end() ? rows_.at(span.seq_id).length() : end->second;
      for (int i = 0; i < span.q_len; ++i, ++want) {
        if (positions[at + i] != want) {
          return false;
        }
      }
      ends[span.seq_id] = want;
      at += span.q_len;
    }
    return true;
  }

  int step_length() const {
    int length = 0;
    for (const Span& span : spans_) {
      length += span.q_len;
    }
    return length;
  }

  int held() const {
    int length = 0;
    for (const auto& row : rows_) {
      length += row.second.length();
    }
    return length;
  }

  bool has_room(int n) const {
    return held() + n <= cfg_.capacity;
  }

  int32_t free_id() const {
    int32_t id = 0;
    while (rows_.find(id) != rows_.end()) {
      ++id;
    }
    return id;
  }

  void invalidate_step() {
    spans_.clear();
    step_pos_.clear();
    declared_ = false;
    placed_ = false;
  }

  static const cache::CacheConfig& checked(const cache::CacheConfig& cfg) {
    if (!cache::valid(cfg)) {
      throw std::runtime_error("MLXBatchedSequenceCache: invalid CacheConfig");
    }
    return cfg;
  }

  cache::CacheConfig cfg_;
  std::map<int32_t, MLXSequenceCache> rows_;
  // Set at init by the delegate that resolved this cache.
  std::optional<::mlx::core::Stream> controller_;
  std::vector<Span> spans_;
  std::vector<int32_t> step_pos_;
  bool declared_ = false;
  bool placed_ = false;
};

} // namespace mlx
} // namespace backends
} // namespace executorch
