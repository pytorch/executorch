/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// The neutral single-sequence controller (SequenceCache) and its flat layout
// policy (FlatPolicy). SequenceCache owns the one logical length for the whole
// model and dispatches per-layer layout to a policy, so a multi-layer model
// stays coherent. Tensor-free / ET-independent.

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

#include <executorch/extension/llm/cache/cache.h>

namespace executorch {
namespace extension {
namespace llm {
namespace cache {

// Full history [0, length): one contiguous write run, read over all history.
class FlatPolicy final : public LayoutPolicy {
 public:
  int retained_from(int /*length*/) const override {
    return 0; // keeps all history
  }
  SeqStepPlan plan(int position, int T) const override {
    const int end = position + T;
    SeqStepPlan p{};
    p.n_write = 1;
    p.write[0] = Run{position, T}; // contiguous append, never wraps
    p.n_read = 1;
    p.read[0] = Run{0, end}; // attend over all history
    p.read_base_pos = 0;
    return p;
  }
};

// One controller for all layers: owns the single logical length, admission, and
// rewind; dispatches per-layer layout to a shared LayoutPolicy. Policies are
// deduped by (kind, window), so a uniform model holds a single policy object.
class SequenceCache : public CacheBase,
                      public SequenceControl,
                      public SequencePlanner {
 public:
  explicit SequenceCache(const CacheConfig& cfg) : capacity_(cfg.capacity) {
    layer_to_policy_.reserve(cfg.n_layers);
    for (int l = 0; l < cfg.n_layers; ++l) {
      // layers size 1 = one config broadcast to every layer, else per-layer.
      const LayerConfig& lc =
          cfg.layers.size() == 1 ? cfg.layers.front() : cfg.layers[l];
      layer_to_policy_.push_back(policy_index(lc.policy));
    }
  }

  // CacheBase: face recovery without RTTI.
  SequenceControl* as_control() override {
    return this;
  }
  SequencePlanner* as_planner() override {
    return this;
  }

  // SequenceControl.
  bool can_extend(int n = 1) const override {
    return length_ + n <=
        capacity_; // evicting layers reuse rows; capacity bounds
  }
  int capacity() const override {
    return capacity_;
  }
  void clear() override {
    length_ = 0;
  }
  bool rewind(int new_len) override {
    if (new_len > length_) {
      return false; // cannot grow
    }
    // An evicting layer physically drops everything older than it retains, so
    // the target must be no older than the most-restrictive layer retains.
    int floor = 0;
    for (const auto& p : policies_) {
      floor = std::max(floor, p->retained_from(length_));
    }
    if (new_len < floor) {
      return false; // history evicted from an evicting layer
    }
    length_ = new_len;
    return true;
  }

  // SequencePlanner. plan() is pure; commit() advances the length.
  std::optional<SeqStepPlan> plan(int layer, int position, int T)
      const override {
    if (layer < 0 || layer >= static_cast<int>(layer_to_policy_.size())) {
      return std::nullopt;
    }
    if (position + T > capacity_) {
      return std::nullopt;
    }
    return policies_[layer_to_policy_[layer]]->plan(position, T);
  }
  void commit(const SeqStepPlan& plan) override {
    // end = read_base_pos + read length (the read spans up to the logical end).
    // Idempotent: commit the max, so one call per step (not per layer)
    // suffices.
    int end = plan.read_base_pos;
    for (int i = 0; i < plan.n_read; ++i) {
      end += plan.read[i].len;
    }
    length_ = std::max(length_, end);
  }

 private:
  int policy_index(const LayerPolicy& lp) {
    for (std::size_t i = 0; i < specs_.size(); ++i) {
      if (specs_[i].kind == lp.kind && specs_[i].window == lp.window) {
        return static_cast<int>(i);
      }
    }
    specs_.push_back(lp);
    policies_.push_back(make_policy(lp));
    return static_cast<int>(policies_.size() - 1);
  }
  std::unique_ptr<LayoutPolicy> make_policy(const LayerPolicy&) const {
    return std::make_unique<FlatPolicy>();
  }

  int capacity_;
  int length_ = 0;
  std::vector<LayerPolicy> specs_; // parallel to policies_, for dedup
  std::vector<std::unique_ptr<LayoutPolicy>> policies_;
  std::vector<int> layer_to_policy_;
};

} // namespace cache
} // namespace llm
} // namespace extension
} // namespace executorch
