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
// no contiguous range and a fork sets a second owner bit. The step's tokens sit
// on one flat axis, sequence identity supplied out-of-band by begin_step.
// Tensor-free; the byte layer holds the pools and applies what it is given.

#include <array>
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
// means the same token in every layer's pool. A fused kind means the step's
// cells are a contiguous run at the tail of a window its sequence owns
// outright, so the write is one run and no mask is needed. Explicit gives each
// token's cell and the bits it may attend.
struct CellStep {
  MaskKind kind;
  int n_tok;
  int read_len; // the window is cells [0, read_len)
  int write_start; // fused only: first cell of the run
  std::vector<int32_t> cells; // Explicit only: cell per query token
  std::vector<uint8_t>
      mask_bits; // Explicit only: [n_tok, read_len], 1 = attend
};

// Backend face of the cell layout. The step's first layer places its tokens and
// every later layer reuses that placement. `layer` selects the window, which
// decides the kind and mask, so a step is per policy and memoized for the
// forward. The returned step is owned by the cache and valid until the next
// verb. nullptr = no declaration, a token count disagreeing with it, a position
// a sequence already holds, a layer out of range, or a layer served twice.
class CellStepper {
 public:
  virtual ~CellStepper() = default;
  virtual const CellStep*
  place_step(int layer, const int32_t* positions, int n_tok) = 0;
};

class CellCache : public CacheBase, public BatchControl, public CellStepper {
 public:
  // One bit per sequence in the owner bitset.
  static constexpr int kMaxSeqs = 64;

  // Precondition: valid(cfg). CacheBuilderRegistry::build enforces it for
  // registry-created caches; direct construction must check first.
  explicit CellCache(const CacheConfig& cfg);

  CacheBase* base() {
    return this;
  }
  BatchControl* as_batch_control() override {
    return this;
  }
  CellStepper* as_cell_stepper() override {
    return this;
  }

  // -- CacheControl ------------------------------------------------------

  bool can_extend(int n = 1) const override;
  int capacity() const override;
  void clear() override;

  // -- BatchControl ------------------------------------------------------

  bool begin_step(const std::vector<int32_t>& seq_ids) override;
  bool seq_cp(int32_t src, int32_t dst, std::optional<int> upto) override;
  bool seq_rm(int32_t seq, int p0, std::optional<int> p1) override;
  int seq_len(int32_t seq) const override;
  int next_pos(int32_t seq) const override;

  int free_cells() const;
  int used_end() const;

  // -- CellStepper -------------------------------------------------------

  const CellStep* place_step(int layer, const int32_t* positions, int n_tok)
      override;

 private:
  struct SeqInfo {
    int count = 0;
    int min_cell = 0;
    int max_cell = -1;
    int max_pos = -1;
  };

  static uint64_t bit(int32_t seq);
  static bool valid_seq(int32_t seq);

  // Every position must be newer than what that sequence already holds, or it
  // would own two cells for one token. Two cells with the same pos and owner
  // are indistinguishable, so a branch is its own sequence.
  bool extends(const int32_t* positions, int n_tok) const;

  // Placement policy: the lowest free cell, so freed cells refill before the
  // extent grows. The choice moves only the read window's width and how often a
  // step fuses; the mask keys off pos/owners, never the index.
  int lowest_free(int from) const;

  void invalidate_steps();
  void invalidate_step();

  // Recompute a sequence's summary after the verbs move cells under it.
  void rescan(int32_t seq);

  void claim(int cell, int32_t pos, int32_t seq);

  // Claim a cell per token, shared by every layer of the forward. False = the
  // pool cannot supply them; no cell is claimed until every one is found, so a
  // refusal leaves the table unchanged.
  bool place();

  // One window's step, built the first time a layer with this window asks and
  // kept for the rest of the step.
  const CellStep& step_for(int window);

  // Fused needs one sequence owning the whole read window, with this step's
  // cells the run at its tail. A sliding window bounds queries from below,
  // which no fused kind expresses.
  bool fused(int window) const;

  // Query i attends cell j iff j is occupied, shares a sequence with i, is no
  // newer than i, and on a windowed layer no older than its window. The step's
  // cells are already placed, so a query sees itself and its earlier tokens.
  std::vector<uint8_t> build_mask(int window) const;

  int capacity_;
  std::vector<int32_t> pos_; // per cell; -1 = free
  std::vector<uint64_t> owners_; // per cell; owning-sequence bitset
  int used_count_ = 0; // occupied cells, so admission stays O(1)
  int used_end_ = 0; // every occupied cell is in [0, used_end)
  std::array<SeqInfo, kMaxSeqs> info_{};

  std::vector<int32_t> step_seq_ids_; // set by begin_step
  std::vector<int32_t> step_pos_; // set when the step is placed
  std::vector<int32_t> cells_; // the step's placement, shared by every layer
  std::vector<bool> served_; // layers this step has already answered
  std::vector<int> windows_; // per layer; 0 = keeps all history
  // window -> step, memoized per forward. Node-based is required: a step
  // handed to one layer must survive another layer's insert.
  std::map<int, CellStep> steps_;
  bool declared_ = false;
  bool placed_ = false;
};

} // namespace cache
} // namespace llm
} // namespace extension
} // namespace executorch
