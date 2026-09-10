/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/cache/batched_sequence_cache.h>

#include <algorithm>

namespace executorch {
namespace extension {
namespace llm {
namespace cache {

BatchedSequenceCache::BatchedSequenceCache(const CacheConfig& cfg)
    : cfg_(cfg) {}

// -- CacheControl -----------------------------------------------------------

int BatchedSequenceCache::capacity() const {
  return cfg_.capacity;
}

void BatchedSequenceCache::clear() {
  rows_.clear();
  invalidate_step();
}

// -- BatchControl -----------------------------------------------------------

bool BatchedSequenceCache::declare_step(const std::vector<int32_t>& seq_ids) {
  invalidate_step();
  if (seq_ids.empty()) {
    return false;
  }
  // The pool is weighed once against the whole step; each sequence's own reach
  // is weighed against the tokens the step gives it.
  if (!has_room(static_cast<int>(seq_ids.size()))) {
    return false;
  }
  std::map<int32_t, int> taking;
  for (int32_t id : seq_ids) {
    if (rows_.find(id) == rows_.end()) {
      return false; // only an id seq_new handed out
    }
    ++taking[id];
  }
  for (const auto& [id, n] : taking) {
    if (!within_context(id, n)) {
      return false;
    }
  }
  // Run-length encode: consecutive tokens of one sequence are one span, and a
  // sequence named twice in a step gets one span each time.
  for (std::size_t i = 0; i < seq_ids.size();) {
    std::size_t j = i + 1;
    while (j < seq_ids.size() && seq_ids[j] == seq_ids[i]) {
      ++j;
    }
    spans_.push_back(SeqSpan{seq_ids[i], static_cast<int>(j - i), {}});
    i = j;
  }
  declared_ = true;
  return true;
}

bool BatchedSequenceCache::can_admit(int32_t seq_id, int n) const {
  return rows_.find(seq_id) != rows_.end() && n >= 0 &&
      within_context(seq_id, n);
}

std::optional<int> BatchedSequenceCache::max_seqs() const {
  return std::nullopt;
}

std::optional<int32_t> BatchedSequenceCache::seq_new() {
  const std::optional<int32_t> id = free_id();
  if (id) {
    rows_.emplace(*id, SequenceCache(cfg_));
  }
  return id;
}

std::optional<int32_t> BatchedSequenceCache::seq_clone(
    int32_t src,
    std::optional<int> upto) {
  auto it = rows_.find(src);
  if (it == rows_.end() || it->second.length() == 0) {
    return std::nullopt;
  }
  const int len =
      upto ? std::min(*upto, it->second.length()) : it->second.length();
  if (len <= 0 || !has_room(len)) {
    return std::nullopt;
  }
  // Copied before rewinding: the floor a windowed layer imposes is a function
  // of the length the source holds now.
  SequenceCache fork(it->second);
  if (!fork.rewind(len)) {
    return std::nullopt; // older than a windowed layer retains
  }
  const std::optional<int32_t> dst = free_id();
  if (!dst) {
    return std::nullopt;
  }
  rows_.emplace(*dst, std::move(fork));
  invalidate_step();
  return dst;
}

bool BatchedSequenceCache::seq_rm(int32_t seq_id) {
  auto it = rows_.find(seq_id);
  if (it == rows_.end()) {
    return false;
  }
  rows_.erase(it);
  invalidate_step();
  return true;
}

bool BatchedSequenceCache::rewind(int32_t seq_id, int new_len) {
  auto it = rows_.find(seq_id);
  if (it == rows_.end() || new_len < 0 || !it->second.rewind(new_len)) {
    return false;
  }
  invalidate_step();
  return true;
}

int BatchedSequenceCache::seq_len(int32_t seq_id) const {
  auto it = rows_.find(seq_id);
  return it == rows_.end() ? 0 : it->second.length();
}

int BatchedSequenceCache::next_pos(int32_t seq_id) const {
  return seq_len(seq_id);
}

// -- SeqSpanStepper ---------------------------------------------------------

const std::vector<SeqSpan>* BatchedSequenceCache::place_step(
    int layer,
    const int32_t* positions,
    int length) {
  if (layer < 0 || layer >= cfg_.n_layers) {
    return nullptr;
  }
  if (placed_) {
    // A repeat within the placed forward. Every layer of a forward carries the
    // same tokens, and a KV-shared layer re-serves its donor's id, so a repeat
    // is answered from the placement already made. Only identity is checked:
    // the rows have advanced, so the positions that placed the step no longer
    // continue them.
    if (length != static_cast<int>(step_pos_.size()) ||
        !std::equal(positions, positions + length, step_pos_.begin())) {
      return nullptr;
    }
  } else {
    int total = 0;
    for (const SeqSpan& span : spans_) {
      total += span.q_len;
    }
    if (!declared_ || length != total || !check_positions(positions)) {
      return nullptr;
    }
    step_pos_.assign(positions, positions + length);
    declared_ = false; // one declaration, one placement
    placed_ = true;
  }
  // plan() reads the position it is handed rather than the row's length, and
  // commit() takes a max, so every layer plans the same step independently and
  // a repeat lands the rows where the placement already left them.
  int at = 0;
  for (SeqSpan& span : spans_) {
    SequenceCache& row = rows_.at(span.seq_id);
    std::optional<SeqStepPlan> p = row.plan(layer, positions[at], span.q_len);
    if (!p) {
      return nullptr;
    }
    span.plan = *p;
    at += span.q_len;
  }
  for (const SeqSpan& span : spans_) {
    rows_.at(span.seq_id).commit(span.plan);
  }
  return &spans_;
}

// -- internals --------------------------------------------------------------

std::optional<int32_t> BatchedSequenceCache::free_id() const {
  int32_t id = 0;
  while (rows_.find(id) != rows_.end()) {
    ++id;
  }
  return id;
}

int BatchedSequenceCache::held() const {
  int n = 0;
  for (const auto& row : rows_) {
    n += row.second.length();
  }
  return n;
}

bool BatchedSequenceCache::has_room(int n) const {
  return held() + n <= cfg_.capacity;
}

bool BatchedSequenceCache::within_context(int32_t seq_id, int n) const {
  auto it = rows_.find(seq_id);
  return it != rows_.end() &&
      (!cfg_.max_context || it->second.length() + n <= *cfg_.max_context);
}

bool BatchedSequenceCache::check_positions(const int32_t* positions) const {
  std::map<int32_t, int> ends;
  int at = 0;
  for (const SeqSpan& span : spans_) {
    auto e = ends.find(span.seq_id);
    int want = e != ends.end() ? e->second : rows_.at(span.seq_id).length();
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

void BatchedSequenceCache::invalidate_step() {
  spans_.clear();
  step_pos_.clear();
  declared_ = false;
  placed_ = false;
}

} // namespace cache
} // namespace llm
} // namespace extension
} // namespace executorch
