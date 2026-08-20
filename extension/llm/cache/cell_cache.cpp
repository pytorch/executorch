/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/cache/cell_cache.h>

#include <algorithm>
#include <cassert>

namespace executorch {
namespace extension {
namespace llm {
namespace cache {

CellCache::CellCache(const CacheConfig& cfg)
    : capacity_(cfg.capacity),
      pos_(cfg.capacity, -1),
      owners_(cfg.capacity, 0),
      served_(cfg.n_layers, false) {
  assert(valid(cfg));
  // One window per layer, from the same per-layer config the sequence cache
  // reads. Layers agreeing on a window share a step.
  windows_.reserve(cfg.n_layers);
  for (int l = 0; l < cfg.n_layers; ++l) {
    const LayerConfig& lc =
        cfg.layers.size() == 1 ? cfg.layers.front() : cfg.layers[l];
    windows_.push_back(
        lc.policy.kind == LayerPolicy::Kind::Ring ? lc.policy.window : 0);
  }
}

// -- CacheControl ------------------------------------------------------------

bool CellCache::can_extend(int n) const {
  return capacity_ - used_count_ >= n;
}

int CellCache::capacity() const {
  return capacity_;
}

void CellCache::clear() {
  std::fill(pos_.begin(), pos_.end(), -1);
  std::fill(owners_.begin(), owners_.end(), 0);
  info_.fill(SeqInfo{});
  used_end_ = 0;
  used_count_ = 0;
  invalidate_step();
}

// -- BatchControl ------------------------------------------------------------

bool CellCache::begin_step(const std::vector<int32_t>& seq_ids) {
  if (seq_ids.empty() || !can_extend(static_cast<int>(seq_ids.size()))) {
    return false;
  }
  for (int32_t seq : seq_ids) {
    if (!valid_seq(seq)) {
      return false;
    }
  }
  step_seq_ids_ = seq_ids;
  declared_ = true;
  invalidate_steps();
  std::fill(served_.begin(), served_.end(), false);
  return true;
}

bool CellCache::seq_cp(int32_t src, int32_t dst, std::optional<int> upto) {
  // A fork targets a fresh sequence: `upto` bounds a prefix, so copying onto
  // a sequence that already holds slots would give it two for one position.
  if (!valid_seq(src) || !valid_seq(dst) || src == dst ||
      info_[dst].count != 0) {
    return false;
  }
  const uint64_t src_bit = bit(src), dst_bit = bit(dst);
  for (int i = 0; i < used_end_; ++i) {
    if ((owners_[i] & src_bit) && (!upto || pos_[i] < *upto)) {
      owners_[i] |= dst_bit;
    }
  }
  rescan(dst);
  invalidate_steps();
  return true;
}

bool CellCache::seq_rm(int32_t seq, int p0, std::optional<int> p1) {
  if (!valid_seq(seq)) {
    return false;
  }
  const uint64_t b = bit(seq);
  for (int i = 0; i < used_end_; ++i) {
    if ((owners_[i] & b) && pos_[i] >= p0 && (!p1 || pos_[i] < *p1)) {
      owners_[i] &= ~b;
      if (owners_[i] == 0) {
        pos_[i] = -1;
        --used_count_;
      }
    }
  }
  while (used_end_ > 0 && pos_[used_end_ - 1] < 0) {
    --used_end_;
  }
  rescan(seq);
  invalidate_steps();
  return true;
}

int CellCache::seq_len(int32_t seq) const {
  return valid_seq(seq) ? info_[seq].count : 0;
}

int CellCache::next_pos(int32_t seq) const {
  return valid_seq(seq) ? info_[seq].max_pos + 1 : 0;
}

int CellCache::free_cells() const {
  return capacity_ - used_count_;
}

int CellCache::used_end() const {
  return used_end_;
}

// -- CellStepper -------------------------------------------------------------

const CellStep*
CellCache::place_step(int layer, const int32_t* positions, int n_tok) {
  if (layer < 0 || layer >= static_cast<int>(windows_.size()) ||
      served_[layer]) {
    return nullptr; // out of range, or a forward that skipped begin_step
  }
  if (!placed_) {
    if (!declared_ || n_tok != static_cast<int>(step_seq_ids_.size())) {
      return nullptr; // no declaration, or a token count disagreeing with it
    }
    if (!extends(positions, n_tok)) {
      return nullptr; // nothing mutated yet, so the step can be re-placed
    }
    step_pos_.assign(positions, positions + n_tok);
    if (!place()) {
      return nullptr;
    }
    declared_ = false; // one declaration, one placement
    placed_ = true;
  }
  served_[layer] = true;
  return &step_for(windows_[layer]);
}

// -- internals ---------------------------------------------------------------

uint64_t CellCache::bit(int32_t seq) {
  return uint64_t{1} << seq;
}

bool CellCache::valid_seq(int32_t seq) {
  return seq >= 0 && seq < kMaxSeqs;
}

bool CellCache::extends(const int32_t* positions, int n_tok) const {
  std::array<int32_t, kMaxSeqs> newest{};
  for (int s = 0; s < kMaxSeqs; ++s) {
    newest[s] = info_[s].max_pos;
  }
  for (int i = 0; i < n_tok; ++i) {
    const int32_t seq = step_seq_ids_[i];
    if (positions[i] <= newest[seq]) {
      return false;
    }
    newest[seq] = positions[i];
  }
  return true;
}

int CellCache::lowest_free(int from) const {
  for (int i = from; i < capacity_; ++i) {
    if (pos_[i] < 0) {
      return i;
    }
  }
  return -1;
}

void CellCache::invalidate_steps() {
  placed_ = false;
  steps_.clear();
}

void CellCache::invalidate_step() {
  invalidate_steps();
  declared_ = false;
  std::fill(served_.begin(), served_.end(), false);
  step_seq_ids_.clear();
  step_pos_.clear();
}

void CellCache::rescan(int32_t seq) {
  const uint64_t b = bit(seq);
  SeqInfo info;
  info.min_cell = capacity_;
  for (int i = 0; i < used_end_; ++i) {
    if (!(owners_[i] & b)) {
      continue;
    }
    info.min_cell = std::min(info.min_cell, i);
    info.max_cell = i;
    info.max_pos = std::max(info.max_pos, pos_[i]);
    ++info.count;
  }
  if (info.count == 0) {
    info = SeqInfo{};
  }
  info_[seq] = info;
}

void CellCache::claim(int cell, int32_t pos, int32_t seq) {
  pos_[cell] = pos;
  owners_[cell] = bit(seq);
  used_end_ = std::max(used_end_, cell + 1);
  ++used_count_;
  SeqInfo& info = info_[seq];
  info.min_cell = info.count == 0 ? cell : std::min(info.min_cell, cell);
  info.max_cell = std::max(info.max_cell, cell);
  info.max_pos = std::max(info.max_pos, pos);
  ++info.count;
}

bool CellCache::place() {
  const int n_tok = static_cast<int>(step_pos_.size());
  cells_.resize(n_tok);
  // A free cell leaves every cell below it occupied, so the next scan resumes
  // past it.
  int from = 0;
  for (int i = 0; i < n_tok; ++i) {
    const int cell = lowest_free(from);
    if (cell < 0) {
      return false;
    }
    cells_[i] = cell;
    from = cell + 1;
  }
  for (int i = 0; i < n_tok; ++i) {
    claim(cells_[i], step_pos_[i], step_seq_ids_[i]);
  }
  return true;
}

const CellStep& CellCache::step_for(int window) {
  auto it = steps_.find(window);
  if (it != steps_.end()) {
    return it->second;
  }
  CellStep step;
  step.n_tok = static_cast<int>(cells_.size());
  step.read_len = used_end_;
  if (fused(window)) {
    step.kind = step.n_tok == 1 ? MaskKind::None : MaskKind::Causal;
    step.write_start = cells_.front();
  } else {
    step.kind = MaskKind::Explicit;
    step.write_start = -1;
    step.cells = cells_;
    step.mask_bits = build_mask(window);
  }
  return steps_.emplace(window, std::move(step)).first->second;
}

bool CellCache::fused(int window) const {
  // The window is measured in positions, so the oldest cell in the read
  // window must fall inside the newest query's window.
  if (window > 0) {
    int32_t oldest = INT32_MAX, newest = INT32_MIN;
    for (int j = 0; j < used_end_; ++j) {
      if (pos_[j] >= 0) {
        oldest = std::min(oldest, pos_[j]);
        newest = std::max(newest, pos_[j]);
      }
    }
    if (oldest <= newest && newest - oldest >= window) {
      return false;
    }
  }
  // More than one sequence: each would need a different subset.
  const int32_t seq = step_seq_ids_.front();
  for (int32_t s : step_seq_ids_) {
    if (s != seq) {
      return false;
    }
  }
  // Holes or another sequence's cells: the window is then not this
  // sequence's causal prefix.
  const SeqInfo& info = info_[seq];
  if (info.count != used_end_) {
    return false;
  }
  // The step's cells are the run at the tail.
  const int n_tok = static_cast<int>(cells_.size());
  for (int i = 0; i < n_tok; ++i) {
    if (cells_[i] != used_end_ - n_tok + i) {
      return false;
    }
  }
  return true;
}

std::vector<uint8_t> CellCache::build_mask(int window) const {
  const int n_tok = static_cast<int>(cells_.size());
  std::vector<uint8_t> bits(
      static_cast<size_t>(n_tok) * static_cast<size_t>(used_end_), 0);
  for (int i = 0; i < n_tok; ++i) {
    const uint64_t tok_bit = bit(step_seq_ids_[i]);
    const int32_t tok_pos = step_pos_[i];
    // A flat layer reaches back to the start; a windowed one to its window.
    const int32_t oldest = window > 0 ? tok_pos - window + 1 : 0;
    uint8_t* row = bits.data() + static_cast<size_t>(i) * used_end_;
    for (int j = 0; j < used_end_; ++j) {
      row[j] =
          (pos_[j] >= 0 && // occupied: a freed cell holds nothing
           (owners_[j] & tok_bit) && // one of this query's sequences
           pos_[j] <= tok_pos && // not the future; <= so a query sees itself
           pos_[j] >= oldest); // within the window
    }
  }
  return bits;
}

} // namespace cache
} // namespace llm
} // namespace extension
} // namespace executorch
