/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// The cell layout: many sequences over one pool of per-token cells. A cell
// holds one token's position and the sequences owning it, so a sequence needs
// no contiguous range and a fork sets a second owner bit rather than copying
// K/V. The step's tokens sit on one flat axis, sequence identity supplied
// out-of-band by begin_step, so one sequence or many is the same graph.
// Tensor-free; the byte layer holds the pools and applies the plan.

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include <executorch/extension/llm/cache/cache.h>

namespace executorch {
namespace extension {
namespace llm {
namespace cache {

// Integer-only handoff to the byte layer, covering the whole forward: a cell
// means the same token in every layer's pool, so placement happens once per
// step. A fused kind (None / Causal) means the step's cells are a contiguous
// run at the tail of a window its sequence owns outright -- one run write, no
// mask. Explicit gives each token's cell and the bits it may attend.
struct CellStepPlan {
  MaskKind kind;
  int n_tok;
  int read_len; // the window is cells [0, read_len)
  int write_start; // fused only: first cell of the run
  std::vector<int32_t> cells; // Explicit only: cell per query token
  std::vector<uint8_t>
      mask_bits; // Explicit only: [n_tok, read_len], 1 = attend
};

// Backend face of the cell layout. The step's first layer places its tokens and
// every later layer reuses that placement, so there is no commit -- placing the
// cells is what commits them. `layer` selects the layer's window, which decides
// its kind and mask, so the plan is per policy and memoized for the step.
// nullptr = no declaration, a token count disagreeing with it, a layer out of
// range, or a layer served twice.
class CellPlanner {
 public:
  virtual ~CellPlanner() = default;
  virtual const CellStepPlan*
  plan(int layer, const int32_t* positions, int n_tok) = 0;
};

class CellCache : public CacheBase, public BatchControl, public CellPlanner {
 public:
  // One bit per sequence in the owner bitset.
  static constexpr int kMaxSeqs = 64;

  explicit CellCache(const CacheConfig& cfg)
      : capacity_(cfg.capacity),
        pos_(cfg.capacity, -1),
        owners_(cfg.capacity, 0),
        served_(cfg.n_layers, false) {
    // One window per layer, from the same per-layer config the sequence cache
    // reads. Layers agreeing on a window share a plan.
    windows_.reserve(cfg.n_layers);
    for (int l = 0; l < cfg.n_layers; ++l) {
      const LayerConfig& lc =
          cfg.layers.size() == 1 ? cfg.layers.front() : cfg.layers[l];
      windows_.push_back(
          lc.policy.kind == LayerPolicy::Kind::Ring ? lc.policy.window : 0);
    }
  }

  CacheBase* base() {
    return this;
  }
  BatchControl* as_batch_control() override {
    return this;
  }
  CellPlanner* as_cell_planner() override {
    return this;
  }

  // -- CacheControl ------------------------------------------------------

  bool can_extend(int n = 1) const override {
    return capacity_ - used_count_ >= n;
  }
  int capacity() const override {
    return capacity_;
  }
  void clear() override {
    std::fill(pos_.begin(), pos_.end(), -1);
    std::fill(owners_.begin(), owners_.end(), 0);
    info_.fill(SeqInfo{});
    used_end_ = 0;
    used_count_ = 0;
    invalidate_step();
  }

  // -- BatchControl ------------------------------------------------------

  bool begin_step(const int32_t* seq_ids, int n_tok) override {
    if (n_tok <= 0 || !can_extend(n_tok)) {
      return false;
    }
    for (int i = 0; i < n_tok; ++i) {
      if (seq_ids[i] < 0 || seq_ids[i] >= kMaxSeqs) {
        return false;
      }
    }
    step_seq_ids_.assign(seq_ids, seq_ids + n_tok);
    declared_ = true;
    invalidate_plan();
    std::fill(served_.begin(), served_.end(), false);
    return true;
  }

  void seq_cp(int32_t src, int32_t dst, std::optional<int> upto) override {
    if (!valid_seq(src) || !valid_seq(dst) || src == dst) {
      return;
    }
    const uint64_t src_bit = bit(src), dst_bit = bit(dst);
    for (int i = 0; i < used_end_; ++i) {
      if ((owners_[i] & src_bit) && (!upto || pos_[i] < *upto)) {
        owners_[i] |= dst_bit;
      }
    }
    rescan(dst);
    invalidate_plan();
  }

  void seq_rm(int32_t seq, int p0, std::optional<int> p1) override {
    if (!valid_seq(seq)) {
      return;
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
    invalidate_plan();
  }

  int seq_len(int32_t seq) const override {
    return valid_seq(seq) ? info_[seq].count : 0;
  }
  int next_pos(int32_t seq) const override {
    return valid_seq(seq) ? info_[seq].max_pos + 1 : 0;
  }

  int free_cells() const {
    return capacity_ - used_count_;
  }
  int used_end() const {
    return used_end_;
  }

  // -- CellPlanner -------------------------------------------------------

  const CellStepPlan* plan(int layer, const int32_t* positions, int n_tok)
      override {
    if (layer < 0 || layer >= static_cast<int>(windows_.size()) ||
        served_[layer]) {
      return nullptr; // out of range, or a forward that skipped begin_step
    }
    if (!placed_) {
      if (!declared_ || n_tok != static_cast<int>(step_seq_ids_.size())) {
        return nullptr; // no declaration, or a token count disagreeing with it
      }
      declared_ = false; // one declaration, one attempt at placing it
      if (!extends(positions, n_tok)) {
        return nullptr;
      }
      step_pos_.assign(positions, positions + n_tok);
      place();
      placed_ = true;
    }
    served_[layer] = true;
    return &plan_for(windows_[layer]);
  }

 private:
  struct SeqInfo {
    int count = 0;
    int min_cell = 0;
    int max_cell = -1;
    int max_pos = -1;
  };

  static uint64_t bit(int32_t seq) {
    return uint64_t{1} << seq;
  }
  static bool valid_seq(int32_t seq) {
    return seq >= 0 && seq < kMaxSeqs;
  }

  // A step only extends its sequences: every position must be newer than what
  // that sequence already holds, or it would own two cells for one token.
  // Siblings at one position are a tree, which two cells with the same pos and
  // owner cannot tell apart; here a branch is its own sequence.
  bool extends(const int32_t* positions, int n_tok) const {
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

  // Placement policy: the lowest free cell, so freed cells refill before the
  // extent grows. The choice changes no result -- the mask keys off pos/owners,
  // never the index -- only the read window's width and how often a step fuses.
  // Precondition: begin_step admitted the step, so a free cell exists.
  int lowest_free() const {
    for (int i = 0; i < capacity_; ++i) {
      if (pos_[i] < 0) {
        return i;
      }
    }
    return -1;
  }

  void invalidate_plan() {
    placed_ = false;
    plans_.clear();
  }
  void invalidate_step() {
    invalidate_plan();
    declared_ = false;
    std::fill(served_.begin(), served_.end(), false);
    step_seq_ids_.clear();
    step_pos_.clear();
  }

  // Recompute a sequence's summary after the verbs move cells under it.
  void rescan(int32_t seq) {
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

  int claim(int32_t pos, int32_t seq) {
    const int cell = lowest_free();
    // Only a slip in used_count_ can get here: begin_step admitted the step,
    // and the step claims exactly the cells it declared.
    assert(cell >= 0);
    pos_[cell] = pos;
    owners_[cell] = bit(seq);
    used_end_ = std::max(used_end_, cell + 1);
    ++used_count_;
    SeqInfo& info = info_[seq];
    info.min_cell = info.count == 0 ? cell : std::min(info.min_cell, cell);
    info.max_cell = std::max(info.max_cell, cell);
    info.max_pos = std::max(info.max_pos, pos);
    ++info.count;
    return cell;
  }

  // The step's cells, shared by every layer: a cell means the same token in
  // each layer's pool, so they are claimed once per forward.
  void place() {
    const int n_tok = static_cast<int>(step_pos_.size());
    cells_.resize(n_tok);
    for (int i = 0; i < n_tok; ++i) {
      cells_[i] = claim(step_pos_[i], step_seq_ids_[i]);
    }
  }

  // The plan for one window, built from that placement the first time a layer
  // with this window asks and kept for the rest of the step.
  const CellStepPlan& plan_for(int window) {
    auto it = plans_.find(window);
    if (it != plans_.end()) {
      return it->second;
    }
    CellStepPlan plan;
    plan.n_tok = static_cast<int>(cells_.size());
    plan.read_len = used_end_;
    if (fused(window)) {
      plan.kind = plan.n_tok == 1 ? MaskKind::None : MaskKind::Causal;
      plan.write_start = cells_.front();
    } else {
      plan.kind = MaskKind::Explicit;
      plan.write_start = -1;
      plan.cells = cells_;
      plan.mask_bits = build_mask(window);
    }
    return plans_.emplace(window, std::move(plan)).first->second;
  }

  // Fused needs one sequence owning the whole read window, with this step's
  // cells the run at its tail: the window is then exactly what these queries
  // may attend. A sliding window also bounds them from below, which no fused
  // kind expresses.
  bool fused(int window) const {
    // The window has outgrown the span, so old cells need excluding.
    if (window > 0 && window < used_end_) {
      return false;
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
    // The step's cells are the run at the tail, which is what lower-right
    // alignment means.
    const int n_tok = static_cast<int>(cells_.size());
    for (int i = 0; i < n_tok; ++i) {
      if (cells_[i] != used_end_ - n_tok + i) {
        return false;
      }
    }
    return true;
  }

  // Query i attends cell j iff j is occupied, shares a sequence with i, is no
  // newer than i, and on a windowed layer no older than its window. The step's
  // cells are already placed, so a query sees itself and its earlier tokens.
  std::vector<uint8_t> build_mask(int window) const {
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

  int capacity_;
  std::vector<int32_t> pos_; // per cell; -1 = free
  std::vector<uint64_t> owners_; // per cell; owning-sequence bitset
  int used_count_ = 0; // occupied cells, so admission stays O(1)
  int used_end_ = 0; // every occupied cell is in [0, used_end)
  std::array<SeqInfo, kMaxSeqs> info_{};

  std::vector<int32_t> step_seq_ids_; // set by begin_step
  std::vector<int32_t> step_pos_; // set by the step's first plan()
  std::vector<int32_t> cells_; // the step's placement, shared by every layer
  std::vector<bool> served_; // layers this step has already answered
  std::vector<int> windows_; // per layer; 0 = keeps all history
  std::map<int, CellStepPlan> plans_; // window -> plan, memoized per step
  bool declared_ = false;
  bool placed_ = false;
};

} // namespace cache
} // namespace llm
} // namespace extension
} // namespace executorch
