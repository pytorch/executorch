/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Neutral, tensor-free KV-cache core shared across backends. The runner and the
// bookkeeping depend only on these integer types; a backend's byte layer (its
// tensor writes/gathers and attention) implements update_and_fetch separately
// over its own tensor type. See extension/llm/cache/rfc for the full design.

#include <stdexcept>
#include <vector>

namespace executorch {
namespace extension {
namespace llm {
namespace cache {

// Runner-facing interface: what every cache exposes to the host generation
// loop. Tensor-free (ints only), backend-independent.
class Cache {
 public:
  virtual ~Cache() = default;
  virtual bool can_extend(int n = 1) const = 0; // admission / hard-stop
  virtual int capacity() const = 0; // logical cap
  virtual void clear() = 0; // reset for reuse
};

// Integer-only handoff from bookkeeping to the backend byte layer: where to
// write this step and how much history to read.
struct StepPlan {
  bool contiguous;
  int valid_len; // contiguous read bound
  int write_start; // contiguous write run start
  int write_len; // contiguous write run length
};

// Contiguous single-sequence bookkeeping: history is a length counter; writes
// append at the current position and reads cover [0, length). Backends inherit
// this for the runner-facing methods and the per-step plan.
class ContiguousBookkeeping : public Cache {
 public:
  explicit ContiguousBookkeeping(int capacity) : capacity_(capacity) {}

  bool can_extend(int n = 1) const override {
    return length_ + n <= capacity_;
  }
  int capacity() const override {
    return capacity_;
  }
  // Rows are overwritten on the next write, so reuse is byte-free.
  void clear() override {
    length_ = 0;
  }

  // Truncate to new_len (no byte work; rows overwritten on next write) for
  // agent backtracking / regeneration.
  void rewind(int new_len) {
    if (new_len > length_) {
      throw std::runtime_error("rewind: cannot grow");
    }
    length_ = new_len;
  }

  // Place T tokens at `position` and return the write run + read bound.
  StepPlan plan(int position, int T) {
    const int end = position + T;
    if (end > capacity_) {
      throw std::runtime_error("cache: exceeds capacity");
    }
    if (end > length_) {
      length_ = end;
    }
    return StepPlan{
        /*contiguous=*/true,
        /*valid_len=*/end,
        /*write_start=*/position,
        /*write_len=*/T};
  }

 private:
  int capacity_;
  int length_ = 0;
};

// Model facts + runtime policy the byte layer sizes its pools from.
// Architecture facts (n_layers/n_kv_heads/head_dim) come from .pte metadata;
// capacity and min_chunk are runtime policy. KV storage dtype joins this with
// the quantized pool; kept tensor-free here so the neutral core carries no
// backend types.
//
// n_kv_heads/head_dim are per-layer so hybrid / mixed-head / MLA models (KV
// dims vary by layer) need no config change; a single-element vector means
// uniform across all layers (the common case).
struct CacheConfig {
  int capacity;
  int n_layers;
  std::vector<int> n_kv_heads;
  std::vector<int> head_dim;
  int min_chunk = 512;
};

} // namespace cache
} // namespace llm
} // namespace extension
} // namespace executorch
