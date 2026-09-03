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
// on one flat axis, sequence identity supplied out-of-band by declare_step.
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
// means the same token in every layer's pool.
struct CellStep {
  int length;
  int read_len; // the window is cells [0, read_len)
  std::vector<int32_t> cells; // cell per query token
  std::vector<uint8_t> mask_bits; // [length, read_len], 1 = attend
};

// Backend face of the cell layout. The step's first layer places its tokens and
// every later layer reuses that placement. `layer` selects the window, which
// decides the kind and mask, so a step is per policy and memoized for the
// forward. The returned step is owned by the cache and valid until the next
// verb. nullptr = no declaration, a token count disagreeing with it, a position
// a sequence already holds, a layer out of range, or a layer served twice.
class CellStepper {
 public:
  static constexpr const char* kFaceName = "et.cache.CellStepper";

  virtual ~CellStepper() = default;
  virtual const CellStep*
  place_step(int layer, const int32_t* positions, int length) = 0;
};

class CellCache : public Cache, public BatchControl, public CellStepper {
 public:
  // One bit per sequence in the owner bitset.
  static constexpr int kMaxSeqs = 64;

  // Precondition: valid(cfg). CacheFactory::build enforces it for
  // registry-created caches; direct construction must check first.
  explicit CellCache(const CacheConfig& cfg);

  Cache* base() {
    return this;
  }
  void* face(FaceId id) override {
    return expose<BatchControl, CellStepper>(this, id);
  }

  // -- CacheControl ------------------------------------------------------

  bool can_extend(int n = 1) const override;
  int capacity() const override;
  void clear() override;

  // -- BatchControl ------------------------------------------------------

  bool declare_step(const std::vector<int32_t>& seq_ids) override;
  std::optional<int32_t> seq_new() override;
  std::optional<int32_t> seq_clone(int32_t src, std::optional<int> upto)
      override;
  bool seq_rm(int32_t seq_id, int p0, std::optional<int> p1) override;
  int seq_len(int32_t seq_id) const override;
  int next_pos(int32_t seq_id) const override;

  int free_cells() const;
  int used_end() const;

  // -- CellStepper -------------------------------------------------------

  const CellStep* place_step(int layer, const int32_t* positions, int length)
      override;

 private:
  struct SeqInfo {
    int count = 0;
    int max_pos = -1;
  };

  static uint64_t bit(int32_t seq_id);
  // Live from the moment seq_new hands the id out until its last slot goes.
  // The bit alone answers this because a sequence can only take slots under an
  // id declare_step accepted, and only a live id is accepted.
  bool live(int32_t seq_id) const;
  static bool valid_seq(int32_t seq_id);

  // Every position must be newer than what that sequence already holds, or it
  // would own two cells for one token. Two cells with the same pos and owner
  // are indistinguishable, so a branch is its own sequence.
  bool extends(const int32_t* positions, int length) const;

  // Placement policy: the lowest free cell, so freed cells refill before the
  // extent grows. The choice moves only the read window's width and how often a
  // step fuses; the mask keys off pos/owners, never the index.
  int lowest_free(int from) const;

  // Drop the step's placement and the per-window steps built from it, after
  // the table moves underneath them.
  void invalidate_steps();

  // Recompute a sequence's summary after the verbs move cells under it.
  void rescan(int32_t seq_id);

  void claim(int cell, int32_t pos, int32_t seq_id);

  // Claim a cell per token, shared by every layer of the forward. False = the
  // pool cannot supply them; no cell is claimed until every one is found, so a
  // refusal leaves the table unchanged.
  bool place();

  // One window's step, built the first time a layer with this window asks and
  // kept for the rest of the step.
  const CellStep& step_for(int window);

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
  uint64_t reserved_ = 0; // ids handed out by seq_new

  std::vector<int32_t> step_seq_ids_; // set by declare_step
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
