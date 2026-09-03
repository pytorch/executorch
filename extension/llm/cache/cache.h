/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Neutral, tensor-free, ET-independent KV-cache core shared across backends. A
// cache exposes a runner-facing control face and a backend-facing planner face,
// recovered from the owning Cache* with as<T>(). Which pair it implements
// depends on the layout: one sequence over per-layer runs, or many sequences
// over a pool of per-token cells.

#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

namespace executorch {
namespace extension {
namespace llm {
namespace cache {

class SequenceControl;
class SequencePlanner;
class BatchControl;
class CellStepper;

// A face is named by a string it declares itself, so a backend can add one
// without this header learning about it. Compared by pointer first, which hits
// whenever both sides share a translation unit or the linker merged the
// literals; the strcmp is the fallback for a cache built in one shared object
// and queried from another.
using FaceId = const char*;

inline bool same_face(FaceId a, FaceId b) {
  return a == b || std::strcmp(a, b) == 0;
}

// Hands back `self` as each face it names, or nullptr for one it does not.
// static_cast does the pointer adjustment a face at a non-zero offset needs,
// refuses to compile if Self does not actually derive from it, and being bound
// to its own name in the pack cannot be paired with the wrong one.
template <class... Fs, class Self>
void* expose(Self* self, FaceId id) {
  void* out = nullptr;
  const bool matched[] = {
      (same_face(id, Fs::kFaceName) ? (out = static_cast<Fs*>(self), true)
                                    : false)...};
  (void)matched;
  return out;
}

// Registry ownership anchor. A cache names the faces it implements from
// face(); everything else it is asked for comes back null.
class Cache {
 public:
  virtual ~Cache() = default;

  // Prefer as<T>(). Overridden with expose<...>(this, id).
  virtual void* face(FaceId) {
    return nullptr;
  }

  // Naming T::kFaceName means a type that is not a face fails to compile,
  // rather than quietly returning null at run time.
  template <class T>
  T* as() {
    return static_cast<T*>(face(T::kFaceName));
  }
};

// Lifecycle and admission, tensor-free.
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
  static constexpr const char* kFaceName = "et.cache.SequenceControl";

  // Truncate to new_len; false = cannot grow, or the target is older than an
  // evicting layer still retains.
  virtual bool rewind(int new_len) = 0;
};

// A contiguous span of physical rows in a layer's pool.
struct Run {
  int start;
  int len;
};

// Integer-only handoff to the backend byte layer. Runs are in logical order
// (oldest -> newest); a flat layer uses one, a ring layer two when it wraps.
// read_base_pos is the logical position of read[0].start.
struct SeqStepPlan {
  Run write[2];
  int n_write;
  Run read[2];
  int n_read;
  int read_base_pos;
};

// Backend face. plan() is const: it computes a layer's layout without changing
// state, and commit() advances the shared logical length. nullopt = the step
// exceeds capacity, or `layer` is out of range.
class SequencePlanner {
 public:
  static constexpr const char* kFaceName = "et.cache.SequencePlanner";

  virtual ~SequencePlanner() = default;
  virtual std::optional<SeqStepPlan> plan(int layer, int position, int T)
      const = 0;
  // Advance the logical length past this step. Idempotent, so once per step
  // suffices.
  virtual void commit(const SeqStepPlan& plan) = 0;
};

// Per-layer layout: flat keeps all history, ring slides a window. Stateless.
class LayoutPolicy {
 public:
  virtual ~LayoutPolicy() = default;
  // Write/read runs for T cells at logical `position`. Precondition: T fits the
  // policy's window.
  virtual SeqStepPlan plan(int position, int T) const = 0;
  // Oldest logical position still retained at this length: 0 for flat,
  // length - window for ring.
  virtual int retained_from(int length) const = 0;
};

// Application face of any multi-sequence cache: the sequence verbs. They run
// between forwards, never during one.
class BatchControl : public CacheControl {
 public:
  static constexpr const char* kFaceName = "et.cache.BatchControl";

  // Which sequence each of the next forward's tokens belongs to, one entry per
  // token; every id must be one seq_new handed out. Also the admission gate:
  // false = rejected and nothing changed, and a step that passes has room for
  // its tokens. Whether its positions are well-formed is checked when the step
  // is placed.
  virtual bool declare_step(const std::vector<int32_t>& seq_ids) = 0;
  // An id no live sequence is using, held until that sequence's last slot is
  // freed. nullopt = every id is in use. Ids may also be chosen by the caller;
  // this only guarantees the one it returns is not already taken.
  virtual std::optional<int32_t> seq_new() = 0;
  // A new sequence claiming src's slots below `upto`, all of them when unset.
  // A shared slot keeps one position, so only a prefix can be shared, and the
  // fork is a snapshot: slots src gains afterwards are its own. Nothing is
  // copied. nullopt = an unknown or empty src, or no free sequence id.
  virtual std::optional<int32_t> seq_clone(
      int32_t src,
      std::optional<int> upto) = 0;
  // Drop the sequence's claim on positions [p0, p1). A slot frees only once
  // no sequence owns it. False = an unknown sequence; a range owning nothing
  // is a no-op.
  virtual bool seq_rm(int32_t seq_id, int p0, std::optional<int> p1) = 0;
  virtual int seq_len(int32_t seq_id) const = 0; // slots the sequence owns
  // one past its newest position
  virtual int next_pos(int32_t seq_id) const = 0;
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

// Model facts and the policy the byte layer sizes its pools from. `layers` is
// per-layer: size 1 applies to every layer, else one entry each.
struct CacheConfig {
  int capacity; // logical cap in cells
  int n_layers;
  std::vector<LayerConfig> layers;
  int kv_dtype; // ET ScalarType the byte layer stores K/V in
  int initial_capacity = 512; // starting pool size; grows lazily to capacity
  // Max tokens per step; a ring layer sizes slots to window + max_write - 1.
  // Unset = each ring layer uses its own window.
  std::optional<int> max_write;
};

// Whether `cfg` satisfies the contract above.
inline bool valid(const CacheConfig& cfg) {
  // initial_capacity may be 0 but not negative, and may exceed capacity -- the
  // byte layer clamps it.
  return cfg.capacity > 0 && cfg.n_layers > 0 && cfg.initial_capacity >= 0 &&
      (cfg.layers.size() == 1 ||
       cfg.layers.size() == static_cast<size_t>(cfg.n_layers));
}

} // namespace cache
} // namespace llm
} // namespace extension
} // namespace executorch
