/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Neutral, tensor-free, ET-independent KV-cache core shared across backends. A
// cache exposes two faces recovered from the owning CacheBase* (static upcasts
// -- no dynamic_cast/RTTI, no diamond): a runner-facing control face and a
// backend-facing planner face. Which pair depends on the kind. A single-
// sequence cache (SequenceCache) drives all layers from one logical length and
// dispatches per-layer layout to a LayoutPolicy (flat = full history, ring =
// sliding window), so a mixed model (e.g. gemma4's alternating full/sliding-
// window layers) stays coherent. A cell cache holds many sequences over one
// pool of per-token cells and plans a whole forward at once. Errors are plain
// C++ (bool / std::optional); cache_et.h adapts them to Error/Result for ET
// consumers.

#include <cstdint>
#include <optional>
#include <vector>

namespace executorch {
namespace extension {
namespace llm {
namespace cache {

class SequenceControl;
class SequencePlanner;
class BatchControl;
class CellPlanner;

// Registry ownership / erasure anchor. The registry owns a cache as a
// CacheBase*; the runner recovers a control face and the backend a planner
// face through these accessors. A cache returns `this` from the pair it
// implements and leaves the other pair null -- static upcasts, no RTTI.
class CacheBase {
 public:
  virtual ~CacheBase() = default;
  virtual SequenceControl* as_control() {
    return nullptr;
  }
  virtual SequencePlanner* as_planner() {
    return nullptr;
  }
  virtual BatchControl* as_batch_control() {
    return nullptr;
  }
  virtual CellPlanner* as_cell_planner() {
    return nullptr;
  }
};

// What every cache exposes to the application: lifecycle + admission,
// tensor-free.
class CacheControl {
 public:
  virtual ~CacheControl() = default;
  virtual bool can_extend(int n = 1) const = 0; // admission / hard-stop
  virtual int capacity() const = 0; // logical cap
  virtual void clear() = 0; // reset for reuse
};

// Application face of a single-sequence cache: one length to rewind.
class SequenceControl : public CacheControl {
 public:
  // Truncate to new_len (agent backtracking); false = cannot grow, or the
  // target is older than an evicting layer still retains.
  virtual bool rewind(int new_len) = 0;
};

// A contiguous span of physical rows in a layer's pool.
struct Run {
  int start;
  int len;
};

// Integer-only handoff from the planner to the backend byte layer. Runs are in
// logical order (oldest -> newest); a flat layer uses one run, a ring layer up
// to two (a write/read that wraps the buffer splits in two). read_base_pos is
// the logical position of read[0].start (0 for flat; the window start for
// ring), so the backend can align RoPE / the attention mask.
struct SeqStepPlan {
  Run write[2];
  int n_write;
  Run read[2];
  int n_read;
  int read_base_pos;
};

// Backend face. plan() is pure -- it computes a layer's layout for a step
// without changing state; commit() advances the shared logical length once the
// step is accepted. `layer` selects the layer's policy. nullopt = the step
// would exceed capacity or `layer` is out of range.
class SequencePlanner {
 public:
  virtual ~SequencePlanner() = default;
  virtual std::optional<SeqStepPlan> plan(int layer, int position, int T)
      const = 0;
  // Advance the logical length past this step. Idempotent (commits the max), so
  // calling it once per step -- not per layer -- suffices.
  virtual void commit(const SeqStepPlan& plan) = 0;
};

// Per-layer layout behavior (flat = full history; ring = sliding window). Pure:
// plan() has no side effects, so the controller (SequenceCache) owns length.
class LayoutPolicy {
 public:
  virtual ~LayoutPolicy() = default;
  // Write/read runs for T cells at logical `position`. Precondition: T fits the
  // policy's window (the runner chunks prefill so a step fits).
  virtual SeqStepPlan plan(int position, int T) const = 0;
  // Oldest logical position still retained given the current length (0 for
  // flat; length - window for ring). Used to bound rewind.
  virtual int retained_from(int length) const = 0;
};

// How a step's queries may attend the window the cache hands back: the cache
// owns the semantic, the backend the mechanism. Mirrors the eager reference.
enum class MaskKind : int {
  None = 0, // one query over a window it may attend entirely
  Causal = 1, // queries at the tail of the window, each seeing up to itself
  Explicit = 2 // anything else: mask_bits says which cells are visible
};

// Application face of any multi-sequence cache, whatever its layout: the
// sequence verbs, integer-only. A sequence exists while some slot lists it as
// an owner -- begin_step or seq_cp brings it into being, its last freed slot
// ends it.
class BatchControl : public CacheControl {
 public:
  // Which sequence each of the next forward's tokens belongs to, one entry per
  // token, and the admission gate: false = rejected, nothing changed. Deciding
  // it here means a step that passes cannot later fail to place.
  virtual bool begin_step(const int32_t* seq_ids, int n_tok) = 0;
  // Give dst a claim on src's slots below `upto` (all of them when unset) -- a
  // fork, zero-copy where sequences share a pool. A shared slot keeps one
  // position, so only a prefix can be shared.
  virtual void seq_cp(int32_t src, int32_t dst, std::optional<int> upto) = 0;
  // Drop seq's claim on positions [p0, p1); a slot frees only once no sequence
  // owns it, so removing a shared range reclaims nothing until the last owner
  // lets go.
  virtual void seq_rm(int32_t seq, int p0, std::optional<int> p1) = 0;
  virtual int seq_len(int32_t seq) const = 0; // slots the sequence owns
  virtual int next_pos(int32_t seq) const = 0; // one past its newest position
};

// Per-layer cache kind and its parameters.
struct LayerPolicy {
  enum class Kind : int {
    Flat = 0,
    Ring = 1
  }; // serialized values: append-only
  Kind kind = Kind::Flat;
  int window = 0; // Ring: sliding-window size; must be 0 when Flat
};

// Per-layer architecture facts + cache policy.
struct LayerConfig {
  LayerPolicy policy; // default Flat
  int n_kv_heads;
  int head_dim;
};

// Model facts + runtime policy the byte layer sizes its pools from. capacity is
// the logical cap; kv_dtype is the ET ScalarType the byte layer stores K/V in;
// initial_capacity tunes the byte layer's lazy-doubling pool; max_write is the
// max tokens written per step (a ring layer sizes its slots to window +
// max_write - 1 so a multi-token step fits); unset means each ring layer uses
// its own window. `layers` is per-layer: size 1 == uniform across all layers,
// else == n_layers.
struct CacheConfig {
  int capacity;
  int n_layers;
  std::vector<LayerConfig> layers;
  int kv_dtype;
  int initial_capacity = 512;
  std::optional<int> max_write;
};

// Whether `cfg` satisfies the contract above. Callers must check this before
// constructing a cache: the `layers` broadcast rule is indexed directly, so a
// list that is neither size 1 nor n_layers reads past the end. Reported as a
// bool rather than thrown, so each backend picks its own failure mode.
inline bool valid(const CacheConfig& cfg) {
  // initial_capacity may be 0 (allocate nothing up front) but not negative, and
  // may exceed capacity -- the byte layer clamps it.
  return cfg.capacity > 0 && cfg.n_layers > 0 && cfg.initial_capacity >= 0 &&
      (cfg.layers.size() == 1 ||
       cfg.layers.size() == static_cast<size_t>(cfg.n_layers));
}

} // namespace cache
} // namespace llm
} // namespace extension
} // namespace executorch
