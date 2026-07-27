/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Neutral, tensor-free, ET-independent KV-cache core shared across backends. A
// cache exposes two faces recovered from the owning CacheBase* via
// as_control()/as_planner() (static upcasts -- no dynamic_cast/RTTI, no
// diamond): a runner-facing control face (SequenceControl) and a backend-facing
// planner face (SequencePlanner). One controller (SequenceCache) drives all
// layers and dispatches per-layer layout to a LayoutPolicy (full-history flat
// today), so a multi-layer model shares one logical length. Errors are plain
// C++ (bool / std::optional); cache_et.h adapts them to Error/Result for ET
// consumers.

#include <optional>
#include <vector>

namespace executorch {
namespace extension {
namespace llm {
namespace cache {

class SequenceControl;
class SequencePlanner;

// Registry ownership / erasure anchor. The registry owns a cache as a
// CacheBase*; the runner recovers the control face and the backend recovers the
// planner face through these accessors (each concrete cache returns `this`).
class CacheBase {
 public:
  virtual ~CacheBase() = default;
  virtual SequenceControl* as_control() = 0;
  virtual SequencePlanner* as_planner() = 0;
};

// Application (runner) face: lifecycle + admission, tensor-free.
class SequenceControl {
 public:
  virtual ~SequenceControl() = default;
  virtual bool can_extend(int n = 1) const = 0; // admission / hard-stop
  virtual int capacity() const = 0; // logical cap
  // Truncate to new_len (agent backtracking); false = cannot grow, or the
  // target is older than an evicting layer still retains.
  virtual bool rewind(int new_len) = 0;
  virtual void clear() = 0; // reset for reuse
};

// A contiguous span of physical rows in a layer's pool.
struct Run {
  int start;
  int len;
};

// Integer-only handoff from the planner to the backend byte layer. Runs are in
// logical order (oldest -> newest); a flat layer uses one run, a windowing
// layer up to two (a write/read that wraps its buffer splits in two).
// read_base_pos is the logical position of read[0].start (0 for flat), so the
// backend can align RoPE / the attention mask.
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

// Per-layer layout behavior (e.g. full-history flat). Pure: plan() has no side
// effects, so the controller (SequenceCache) owns length.
class LayoutPolicy {
 public:
  virtual ~LayoutPolicy() = default;
  // Write/read runs for T cells at logical `position`. Precondition: T fits the
  // policy's window (the runner chunks prefill so a step fits).
  virtual SeqStepPlan plan(int position, int T) const = 0;
  // Oldest logical position still retained given the current length (0 for a
  // full-history policy; a windowing policy retains only its last window). Used
  // to bound rewind.
  virtual int retained_from(int length) const = 0;
};

// Per-layer cache kind and its parameters.
struct LayerPolicy {
  enum class Kind : int { Flat = 0 }; // serialized values: append-only
  Kind kind = Kind::Flat;
  int window =
      0; // sliding-window size for windowing policies; 0 = full history
};

// Per-layer architecture facts + cache policy.
struct LayerConfig {
  LayerPolicy policy; // default Flat
  int n_kv_heads;
  int head_dim;
};

// Model facts + runtime policy the byte layer sizes its pools from. capacity is
// the logical cap; initial_capacity tunes the byte layer's lazy-doubling pool.
// `layers` is per-layer: size 1 == uniform across all layers, else == n_layers.
struct CacheConfig {
  int capacity;
  int n_layers;
  std::vector<LayerConfig> layers;
  int initial_capacity = 512;
};

} // namespace cache
} // namespace llm
} // namespace extension
} // namespace executorch
