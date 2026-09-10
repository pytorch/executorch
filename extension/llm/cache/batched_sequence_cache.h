/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Many sequences, each over its own private history, sharing one capacity.
// Tensor-free / ET-independent.
//
// A step arrives flat on the token axis and is answered per sequence: the
// declared ids are run-length encoded into spans, and each span is planned
// against that sequence's own SequenceCache. No sequence appears in another's
// window, so nothing here builds a mask.
//
// The backend-facing stepper face lives here rather than in cache.h: only this
// layout implements it, and only a byte layer that already includes this header
// calls it.

#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include <executorch/extension/llm/cache/cache.h>
#include <executorch/extension/llm/cache/sequence_cache.h>

namespace executorch {
namespace extension {
namespace llm {
namespace cache {

// One sequence's slice of a step, and where its rows go on this layer. The
// spans cover the step's tokens in order, so `q_len` alone places each.
struct ET_EXPERIMENTAL SeqSpan {
  int32_t seq_id;
  int q_len; // tokens this span carries, following the span before it
  SeqStepPlan plan; // that sequence's runs for the layer asked about
};

// Backend face. The spans are owned by the cache and valid until the next verb.
// A layer may be served more than once per forward -- a KV-shared layer
// re-serves its donor's id -- provided the repeat passes the same positions; it
// plans the same runs and advances nothing. nullptr = no declaration, a token
// count disagreeing with it, a position a sequence does not continue, a layer
// out of range, or a re-serve whose positions differ.
class ET_EXPERIMENTAL SeqSpanStepper {
 public:
  static constexpr const char* kFaceName = "et.cache.SeqSpanStepper";

  virtual ~SeqSpanStepper() = default;
  virtual const std::vector<SeqSpan>*
  place_step(int layer, const int32_t* positions, int length) = 0;
};

class ET_EXPERIMENTAL BatchedSequenceCache : public Cache,
                                             public BatchControl,
                                             public SeqSpanStepper {
 public:
  // Precondition: valid(cfg). CacheFactory::build enforces it for
  // registry-created caches; direct construction must check first.
  explicit BatchedSequenceCache(const CacheConfig& cfg);

  // -- CacheControl -------------------------------------------------------

  // capacity bounds the whole cache: the private histories share it.
  int capacity() const override;
  void clear() override;

  // -- BatchControl -------------------------------------------------------

  bool declare_step(const std::vector<int32_t>& seq_ids) override;

  // A row is dense from 0, so its length is also where it has reached.
  bool can_admit(int32_t seq_id, int n = 1) const override;

  // nullopt: a sequence is a map entry, so only the capacity they share and the
  // ids an int32 holds bound them.
  std::optional<int> max_seqs() const override;

  // The lowest id not in use, so an id frees for reuse when its sequence goes.
  std::optional<int32_t> seq_new() override;

  // A fork of src's first `upto` positions, all of them when unset. The prefix
  // is copied, so each fork holds its own and is a snapshot: positions src
  // gains afterwards are its own. nullopt = an unknown or empty src, no room
  // for the copy, or a prefix a windowed layer no longer retains.
  //
  // final: the bookkeeping here and the cells a byte layer holds must move
  // together, so the byte half is a hook rather than an override to remember.
  std::optional<int32_t> seq_clone(int32_t src, std::optional<int> upto) final;

  bool seq_rm(int32_t seq_id) override;

  // A windowed layer has physically dropped what it no longer retains, so a
  // target older than that is refused.
  bool rewind(int32_t seq_id, int new_len) override;

  // Dense from 0, so a length and a next position are the same number: every
  // span continues its sequence and no prefix can be removed.
  int seq_len(int32_t seq_id) const override;
  int next_pos(int32_t seq_id) const override;

  // -- SeqSpanStepper -----------------------------------------------------

  const std::vector<SeqSpan>*
  place_step(int layer, const int32_t* positions, int length) override;

 protected:
  // Put src's cells under dst. Called once seq_clone has taken the fork, so
  // src is present and dst is new. The fork's bookkeeping already claims src's
  // history: a layer that leaves the cells behind reads whatever its own fresh
  // storage holds, which is not an error anything downstream can detect.
  virtual void clone_bytes(int32_t /*src*/, int32_t /*dst*/) {}

  void* face(FaceId id) override {
    return expose<BatchControl, SeqSpanStepper>(this, id);
  }

 private:
  // The lowest id not in use. Always finds one: ids are map keys, so a gap
  // exists below rows_.size() however many are live.
  int32_t free_id() const;

  int held() const;

  // Room for n more tokens, whoever they belong to.
  bool has_room(int n) const;
  bool within_context(int32_t seq_id, int n) const;

  // Every span continues its own sequence: a consecutive run from where that
  // sequence ends. A sequence spanned twice continues across both.
  bool check_positions(const int32_t* positions) const;

  void invalidate_step();

  CacheConfig cfg_;
  std::map<int32_t, SequenceCache> rows_;
  std::vector<SeqSpan> spans_; // this step's, shared by every layer
  std::vector<int32_t> step_pos_; // the placed step, for a repeat to match
  bool declared_ = false;
  bool placed_ = false;
};

} // namespace cache
} // namespace llm
} // namespace extension
} // namespace executorch
