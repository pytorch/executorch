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
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "MLXCache.h"
#include "MLXSequenceCacheV2.h"

#include <executorch/extension/llm/cache/cache.h>

namespace executorch {
namespace backends {
namespace mlx {

namespace cache = ::executorch::extension::llm::cache;

// A private MLXSequenceCacheV2 per sequence, sharing one logical token budget.
// Packed spans attend over their sequence's history and rejoin in token order.
// Interleaved spans of one owner are grouped into a single V2 call per layer.
// IDs and positions cover K/V. Q follows the global packed order of the query
// ranges reported by each sequence. V2 owns admission and input interpretation.
// Inputs must be MLX-managed; execution remains entirely caller-owned. Controls
// run between forwards. A failing later row may leave earlier rows updated;
// clear() recovers from failed execution. Historical updates are
// layer-specific.
class MLXBatchedCache : public cache::Cache,
                        public cache::BatchControl,
                        public MLXCache {
 public:
  MLXBatchedCache(
      const cache::CacheGeometry& geometry,
      const cache::CacheConfig& cfg)
      : geometry_(geometry), cfg_(cfg) {
    // Validate creation parameters at the sequence boundary.
    (void)MLXSequenceCacheV2(geometry_, cfg_);
  }

  // Total persistent logical tokens across all sequences, including forks.
  int capacity() const override {
    return cfg_.capacity;
  }

  void clear() override {
    rows_.clear();
    invalidate_step();
  }

  bool declare_step(const std::vector<int32_t>& seq_ids) override {
    if (seq_ids.empty() ||
        seq_ids.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
      return false;
    }
    for (int32_t id : seq_ids) {
      if (rows_.find(id) == rows_.end()) {
        return false;
      }
    }

    std::vector<Span> spans;
    std::vector<std::vector<size_t>> groups;
    std::map<int32_t, size_t> group_ids;
    for (size_t i = 0; i < seq_ids.size();) {
      size_t j = i + 1;
      while (j < seq_ids.size() && seq_ids[j] == seq_ids[i]) {
        ++j;
      }
      const auto entry = group_ids.emplace(seq_ids[i], groups.size());
      if (entry.second) {
        groups.emplace_back();
      }
      auto& group = groups[entry.first->second];
      const int offset = group.empty()
          ? 0
          : spans[group.back()].group_at + spans[group.back()].q_len;
      group.push_back(spans.size());
      spans.push_back(
          {seq_ids[i],
           static_cast<int>(j - i),
           static_cast<int>(i),
           entry.first->second,
           offset});
      i = j;
    }
    int64_t required = reserved_tokens();
    for (const auto& group : groups) {
      const Span& last = spans[group.back()];
      const auto growth =
          rows_.at(last.seq_id).admission_growth(last.group_at + last.q_len);
      if (!growth || (required += *growth) > cfg_.capacity) {
        return false;
      }
    }
    invalidate_step();
    spans_ = std::move(spans);
    groups_ = std::move(groups);
    declared_ = true;
    return true;
  }

  std::optional<int> max_seqs() const override {
    return std::nullopt;
  }

  std::optional<int32_t> seq_new() override {
    const auto id = free_id();
    if (id) {
      rows_.try_emplace(*id, geometry_, cfg_);
    }
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
    if (keep <= 0 || !it->second.can_fork(keep) ||
        reserved_tokens() + keep > cfg_.capacity) {
      return std::nullopt;
    }

    const auto dst = free_id();
    if (!dst) {
      return std::nullopt;
    }
    rows_.try_emplace(*dst, it->second, keep);
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

  bool can_rewind(int32_t seq_id, int position) const {
    auto it = rows_.find(seq_id);
    return it != rows_.end() && position >= 0 &&
        it->second.can_rewind(position);
  }

  bool rewind(int32_t seq_id, int position) override {
    if (!can_rewind(seq_id, position) || !rows_.at(seq_id).rewind(position)) {
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
    validate_step(positions, q, k, v);
    const auto query_spans = make_query_spans(layer, q.shape(2));
    const auto& queries = query_spans.empty() ? spans_ : query_spans;

    if (!placed_) {
      step_pos_ = positions;
      declared_ = false;
      placed_ = true;
    }

    std::vector<Tensor> outs;
    outs.reserve(groups_.size());
    for (const auto& group : groups_) {
      const Span& first = spans_[group.front()];
      std::vector<int32_t> span_positions;
      const Span& last = spans_[group.back()];
      span_positions.reserve(static_cast<size_t>(last.group_at + last.q_len));
      for (size_t index : group) {
        const Span& span = spans_[index];
        span_positions.insert(
            span_positions.end(),
            positions.begin() + span.at,
            positions.begin() + span.at + span.q_len);
      }
      outs.push_back(rows_.at(first.seq_id)
                         .attend(
                             layer,
                             span_positions,
                             gather(q, group, queries, s),
                             gather(k, group, spans_, s),
                             gather(v, group, spans_, s),
                             scale,
                             s));
    }
    if (groups_.size() != spans_.size()) {
      std::vector<Tensor> ordered;
      ordered.reserve(spans_.size());
      for (const Span& span : queries) {
        if (span.q_len > 0) {
          ordered.push_back(
              tokens(outs[span.group], span.group_at, span.q_len, s));
        }
      }
      return ordered.size() == 1 ? std::move(ordered.front())
                                 : ::mlx::core::concatenate(ordered, 2, s);
    }
    return outs.size() == 1 ? std::move(outs.front())
                            : ::mlx::core::concatenate(outs, 2, s);
  }

 protected:
  void* face(cache::FaceId id) override {
    return cache::expose<cache::BatchControl, MLXCache>(this, id);
  }

 private:
  struct Span {
    int32_t seq_id;
    int q_len, at;
    size_t group;
    int group_at;
  };

  static Tensor tokens(const Tensor& t, int off, int len, StreamOrDevice s) {
    if (off == 0 && len == t.shape(2)) {
      return t;
    }
    return ::mlx::core::slice(
        t,
        ::mlx::core::Shape{0, 0, off, 0},
        ::mlx::core::Shape{t.shape(0), t.shape(1), off + len, t.shape(3)},
        s);
  }

  Tensor gather(
      const Tensor& t,
      const std::vector<size_t>& group,
      const std::vector<Span>& layout,
      StreamOrDevice s) const {
    const Span* first = nullptr;
    size_t count = 0;
    for (size_t index : group) {
      if (layout[index].q_len > 0) {
        first = &layout[index];
        ++count;
      }
    }
    if (count == 0) {
      throw std::invalid_argument("attend: empty owner span");
    }
    if (count == 1) {
      return tokens(t, first->at, first->q_len, s);
    }
    std::vector<Tensor> parts;
    parts.reserve(count);
    for (size_t index : group) {
      const Span& span = layout[index];
      if (span.q_len > 0) {
        parts.push_back(tokens(t, span.at, span.q_len, s));
      }
    }
    return ::mlx::core::concatenate(parts, 2, s);
  }

  std::vector<Span> make_query_spans(int layer, int query_tokens) const {
    std::vector<Span> queries;
    int64_t expected = 0;
    for (const auto& group : groups_) {
      const Span& last = spans_[group.back()];
      const int n = last.group_at + last.q_len;
      const auto range = rows_.at(last.seq_id).query_span(layer, n);
      expected += range.count;
      // Reuse the K/V layout when all query ranges cover their entire owner.
      if (queries.empty() && (range.begin != 0 || range.count != n)) {
        queries = spans_;
      }
      if (!queries.empty()) {
        for (size_t index : group) {
          auto& span = queries[index];
          const int begin = std::max(span.group_at, range.begin);
          const int end =
              std::min(span.group_at + span.q_len, range.begin + range.count);
          span.q_len = std::max(0, end - begin);
          span.group_at = span.q_len > 0 ? begin - range.begin : 0;
        }
      }
    }
    if (query_tokens != expected) {
      throw std::invalid_argument(
          "attend: packed query count differs from sequence layout");
    }
    // Pack Q in original K/V span order, not first-owner appearance order.
    int at = 0;
    for (auto& span : queries) {
      span.at = at;
      at += span.q_len;
    }
    return queries;
  }

  void validate_step(
      const std::vector<int32_t>& positions,
      const Tensor& q,
      const Tensor& k,
      const Tensor& v) const {
    if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
      throw std::runtime_error("attend: Q/K/V must be BHSD tensors");
    }
    if (positions.size() >
        static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("attend: too many positions");
    }
    const int length = static_cast<int>(positions.size());
    if (k.shape(2) != length || v.shape(2) != length) {
      throw std::runtime_error(
          "attend: one position per key/value token expected");
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
  }

  bool check_positions(const std::vector<int32_t>& positions) const {
    std::map<int32_t, int64_t> ends;
    size_t at = 0;
    for (const Span& span : spans_) {
      auto end = ends.find(span.seq_id);
      int64_t want =
          end == ends.end() ? rows_.at(span.seq_id).length() : end->second;
      for (int i = 0; i < span.q_len; ++i, ++want) {
        if (positions[at++] != want) {
          return false;
        }
      }
      ends[span.seq_id] = want;
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

  int64_t reserved_tokens() const {
    int64_t total = 0;
    for (const auto& row : rows_) {
      total += row.second.reserved_length();
    }
    return total;
  }

  std::optional<int32_t> free_id() const {
    int32_t id = 0;
    while (rows_.find(id) != rows_.end()) {
      if (id == std::numeric_limits<int32_t>::max()) {
        return std::nullopt;
      }
      ++id;
    }
    return id;
  }

  void invalidate_step() {
    spans_.clear();
    groups_.clear();
    step_pos_.clear();
    declared_ = false;
    placed_ = false;
  }

  cache::CacheGeometry geometry_;
  cache::CacheConfig cfg_;
  std::map<int32_t, MLXSequenceCacheV2> rows_;
  std::vector<Span> spans_;
  std::vector<std::vector<size_t>> groups_;
  std::vector<int32_t> step_pos_;
  bool declared_ = false;
  bool placed_ = false;
};

} // namespace mlx
} // namespace backends
} // namespace executorch
