/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// ExecuTorch adapter for the neutral cache core. The core (cache.h /
// sequence_cache.h) is ET-independent and reports failures as
// bool/std::optional so it is usable outside an ET runner. These thin inline
// adapters map those results to ExecuTorch Error/Result (logging on failure)
// for ET consumers -- the runner and the delegate byte layer. (The registry is
// delegate-specific and already returns Result directly, so it needs no
// adapter.)

#include <optional>
#include <vector>

#include <executorch/extension/llm/cache/cache.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

namespace executorch {
namespace extension {
namespace llm {
namespace cache {
namespace et {

using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

// Plan a layer's step, or OutOfResources if it would exceed capacity (or the
// layer is out of range).
inline Result<SeqStepPlan>
plan(const SequencePlanner& planner, int layer, int position, int T) {
  std::optional<SeqStepPlan> p = planner.plan(layer, position, T);
  ET_CHECK_OR_RETURN_ERROR(
      p.has_value(),
      OutOfResources,
      "cache: plan(layer=%d, position=%d, T=%d) exceeds capacity or bad layer",
      layer,
      position,
      T);
  return *p;
}

// Truncate the history, or InvalidArgument if new_len would grow it (or is
// older than an evicting layer retains).
inline Error rewind(SequenceControl& control, int new_len) {
  ET_CHECK_OR_RETURN_ERROR(
      control.rewind(new_len),
      InvalidArgument,
      "rewind: cannot grow to %d",
      new_len);
  return Error::Ok;
}

// A CacheConfig's geometry is a property of the model, not a choice: how many
// caches, and each one's heads, head dim, and attention window. An off-graph
// export publishes those as constant methods, which carry no delegate, so
// reading them needs only the program loaded.
//
// The sizing left unset -- capacity, kv_dtype, initial_capacity -- is the
// caller's policy, and the same program runs under any of it.
//
// InvalidArgument if the program publishes no layout, so it is not an
// off-graph model.
inline Result<CacheConfig> config_from_program(Module& module) {
  const auto read_int = [&module](const char* name) -> std::optional<int64_t> {
    const auto r = module.execute(name);
    if (!r.ok() || r->empty() || !r->at(0).isInt()) {
      return std::nullopt;
    }
    return r->at(0).toInt();
  };
  const auto read_ints =
      [&module](const char* name) -> std::optional<std::vector<int>> {
    const auto r = module.execute(name);
    if (!r.ok() || r->empty() || !r->at(0).isTensor()) {
      return std::nullopt;
    }
    const auto t = r->at(0).toTensor();
    if (t.scalar_type() != ::executorch::aten::ScalarType::Int) {
      return std::nullopt;
    }
    const int32_t* p = t.const_data_ptr<int32_t>();
    return std::vector<int>(p, p + t.numel());
  };

  const auto n_caches = read_int("get_n_caches");
  const auto kv_heads = read_ints("get_kv_heads");
  const auto head_dims = read_ints("get_head_dims");
  const auto windows = read_ints("get_windows");
  ET_CHECK_OR_RETURN_ERROR(
      n_caches && kv_heads && head_dims && windows,
      InvalidArgument,
      "cache: the program publishes no KV layout");
  const auto n = static_cast<size_t>(*n_caches);
  ET_CHECK_OR_RETURN_ERROR(
      kv_heads->size() == n && head_dims->size() == n && windows->size() == n,
      InvalidArgument,
      "cache: the published KV layout names %zu caches inconsistently",
      n);

  CacheConfig cfg{};
  cfg.n_layers = static_cast<int>(n);
  cfg.layers.reserve(n);
  for (size_t l = 0; l < n; ++l) {
    LayerConfig lc{};
    lc.n_kv_heads = (*kv_heads)[l];
    lc.head_dim = (*head_dims)[l];
    lc.policy = (*windows)[l] > 0
        ? LayerPolicy{LayerPolicy::Kind::Ring, (*windows)[l]}
        : LayerPolicy{LayerPolicy::Kind::Flat, 0};
    cfg.layers.push_back(lc);
  }
  return cfg;
}

} // namespace et
} // namespace cache
} // namespace llm
} // namespace extension
} // namespace executorch
